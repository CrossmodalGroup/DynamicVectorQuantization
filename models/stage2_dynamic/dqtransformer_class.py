import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import trange
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence
import numpy as np

import os, sys
sys.path.append(os.getcwd())

from utils.utils import instantiate_from_config
from models.stage2.utils import learning_rate_schedule, disabled_train, top_k_logits, top_p_logits

class Dualformer(pl.LightningModule):
    def __init__(self,
                 transformer_config,
                 first_stage_config,
                 class_cond_stage_config,
                 permuter_config = None,
                 content_loss_weight = 1.0,
                 position_loss_weight = 1.0,
                 activate_sos_for_fine_sequence = True,
                 weight_decay = 0.01,
                 warmup_epochs = 0,
                 monitor = None,
                 ckpt_path = None,
                 ignore_keys = [],
                 ):
        super().__init__()
        self.first_stage_key, self.cond_stage_key = "image", "class_label"
        self.content_loss_weight = content_loss_weight
        self.position_loss_weight = position_loss_weight

        self.init_first_stage_from_ckpt(first_stage_config)
        self.permuter = instantiate_from_config(config=permuter_config)
        self.transformer = instantiate_from_config(config=transformer_config)
        self.cond_stage_model = instantiate_from_config(config=class_cond_stage_config)

        # setting for training scheme
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs

        if monitor is not None:
            self.monitor = monitor
        
        # custom hyperparameters
        self.activate_sos_for_fine_sequence = activate_sos_for_fine_sequence
        self.activate_segment = True if transformer_config["params"]["segment_size"] > 0 else False

        self.content_pad_code = permuter_config["params"]["content_pad_code"]
        self.content_eos_code = permuter_config["params"]["content_eos_code"]
        # self.content_sos_code = class_cond_stage_config["params"]["coarse_sos"] # 

        self.fine_position_sos_code = class_cond_stage_config["params"]["coarse_pos_sos"] # 
        self.coarse_position_eos_code = permuter_config["params"]["coarse_position_eos_code"]
        self.coarse_position_pad_code = permuter_config["params"]["coarse_position_pad_code"]
        self.fine_position_sos_code = class_cond_stage_config["params"]["fine_pos_sos"] # 
        self.fine_position_eos_code = permuter_config["params"]["fine_position_eos_code"]
        self.fine_position_pad_code = permuter_config["params"]["fine_position_pad_code"]

        coarse_hw = permuter_config["params"]["coarse_hw"]
        fine_hw = permuter_config["params"]["fine_hw"]
        self.hw1 = coarse_hw
        self.hw2 = fine_hw // coarse_hw
        self.fine_hw = fine_hw 
        self.fine_position_order = permuter_config["params"]["fine_position_order"]
        self.max_coarse_postion_idx = int(coarse_hw ** 2) - 1
        self.fine_position_eos_tensor = self.permuter.fine_position_eos_tensor.clone()
        self.position_sequence_fine = self.permuter.position_sequence_fine.clone()

        # load pretrained weight
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
    
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
    
    def init_first_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.first_stage_model = model
    
    def configure_optimizers(self):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))
        
        warmup_steps = self.steps_per_epoch * self.warmup_epochs
        multipler_min = self.min_learning_rate / self.learning_rate
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                learning_rate_schedule(
                    warmup_steps=warmup_steps, max_steps=self.training_steps, multipler_min=multipler_min
                ),
            ),
            "interval": "step",
            "frequency": 1,
        }
        
        return [optimizer], [scheduler]
    
    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        if x.size(1) != 3 and len(x.shape) == 4:
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def get_xc(self, batch, N=None):
        x = self.get_input(batch, self.first_stage_key)
        c = batch[self.cond_stage_key]

        if N is not None:
            x = x[:N]
            c = c[:N]
        return x, c
    
    @torch.no_grad()
    def encode_to_c(self, c):
        c_coarse, c_fine, c_pos_coarse, c_pos_fine, c_seg_coarse, c_seg_fine = self.cond_stage_model.encode(c)
        return c_coarse, c_fine, c_pos_coarse, c_pos_fine, c_seg_coarse, c_seg_fine

    @torch.no_grad()
    def encode_to_z(self, x):
        quant, emb_loss, info, grain_indices, gate = self.first_stage_model.encode(x)
        indices = info[2]
        permuted_out = self.permuter(indices=indices, grain_indices=grain_indices)
        return quant, permuted_out
    
    @torch.no_grad()
    def decode_to_img(self, coarse_content, fine_content, coarse_position, fine_position):
        reproduced_indices = self.permuter.forward_back(coarse_content, fine_content, coarse_position, fine_position)
        reproduced_quant, _ = self.first_stage_model.get_code_emb_with_depth(reproduced_indices)
        reproduced_rec = self.first_stage_model.decode(reproduced_quant.permute(0, 3, 1, 2))
        return reproduced_rec
    
    def forward(self, x, c):
        # one step to produce the logits
        _, z_out = self.encode_to_z(x)
        c_coarse, c_fine, c_pos_coarse, c_pos_fine, c_seg_coarse, c_seg_fine = self.encode_to_c(c)

        coarse_content = z_out["coarse_content"]
        fine_content = z_out["fine_content"]
        coarse_position = z_out["coarse_position"]
        fine_position = z_out["fine_position"]
        coarse_segment = z_out["coarse_segment"]
        fine_segment = z_out["fine_segment"]

        az_coarse_content = torch.cat([c_coarse, coarse_content], dim=1)
        az_coarse_position = torch.cat([c_pos_coarse, coarse_position], dim=1)
        az_coarse_segment = torch.cat([c_seg_coarse, coarse_segment], dim=1)
        if self.activate_sos_for_fine_sequence:
            az_fine_content = torch.cat([c_fine, fine_content], dim=1)
            az_fine_position = torch.cat([c_pos_fine, fine_position], dim=1)
            az_fine_segment = torch.cat([c_seg_fine, fine_segment], dim=1)
        else:
            az_fine_content = fine_content
            az_fine_position = fine_position
            az_fine_segment = fine_segment
        
        content_target = torch.cat([az_coarse_content, az_fine_content], dim=1)[:, 1:]
        coarse_position_target = az_coarse_position[:, 1:]
        fine_position_target = az_fine_position

        output = self.transformer(
            coarse_content = az_coarse_content, fine_content = az_fine_content, 
            coarse_position = az_coarse_position, fine_position = az_fine_position, 
            coarse_seg = az_coarse_segment, fine_seg = az_fine_segment, 
            content_target = content_target, coarse_position_target=coarse_position_target, fine_position_target=fine_position_target
        )

        return output
    
    def shared_step(self, batch, batch_idx):
        x, c = self.get_xc(batch)
        output = self(x, c)
        return output
    
    def training_step(self, batch, batch_idx):
        output = self.shared_step(batch, batch_idx)
        position_loss = output["position_loss"]
        content_loss = output["content_loss"]

        total_loss = self.content_loss_weight * content_loss + self.position_loss_weight * position_loss

        self.log("train_content_loss", content_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train_position_loss", position_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train_coarse_position_loss", output["coarse_position_loss"], prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("train_fine_position_loss", output["fine_position_loss"], prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("train_loss", total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        output = self.shared_step(batch, batch_idx)
        position_loss = output["position_loss"]
        content_loss = output["content_loss"]

        total_loss = self.content_loss_weight * content_loss + self.position_loss_weight * position_loss

        self.log("val_content_loss", content_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("val_position_loss", position_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("val_coarse_position_loss", output["coarse_position_loss"], prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("val_fine_position_loss", output["fine_position_loss"], prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("val_loss", total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return total_loss
    
    @torch.no_grad()
    def log_images(self, batch, temperature=None, top_k=None, top_p=None, top_k_pos=None, top_p_pos=None, **kwargs):
        log = dict()

        N = 4
        x, c = self.get_xc(batch, N)
        x = x.to(device=self.device)
        c = c.to(device=self.device)

        c_coarse, c_fine, c_pos_coarse, c_pos_fine, c_seg_coarse, c_seg_fine = self.encode_to_c(c)

        coarse_content, fine_content, coarse_position, fine_position = self.sample_from_scratch(
            c_coarse, c_fine, c_pos_coarse, c_pos_fine, c_seg_coarse, c_seg_fine, 
            temperature = temperature if temperature is not None else 1.0,
            sample = True,
            top_k = top_k if top_k is not None else 300,
            top_p = top_p if top_p is not None else 1.0,
            top_k_pos = top_k_pos if top_k_pos is not None else 100,
            top_p_pos = top_p_pos if top_p_pos is not None else 1.0,
            process = True,
            fix_fine_position = True,
        )
        log["samples_fixed_fine_position"] = self.decode_to_img(coarse_content, fine_content, coarse_position, fine_position)

        coarse_content, fine_content, coarse_position, fine_position = self.sample_from_scratch(
            c_coarse, c_fine, c_pos_coarse, c_pos_fine, c_seg_coarse, c_seg_fine, 
            temperature = temperature if temperature is not None else 1.0,
            sample = True,
            top_k = top_k if top_k is not None else 300,
            top_p = top_p if top_p is not None else 1.0,
            top_k_pos = top_k_pos if top_k_pos is not None else 100,
            top_p_pos = top_p_pos if top_p_pos is not None else 1.0,
            process = True,
            fix_fine_position = False,
        )
        log["samples_from_scratch"] = self.decode_to_img(coarse_content, fine_content, coarse_position, fine_position)

        if self.current_epoch == 0:
            # reconstruction
            _, z_out = self.encode_to_z(x)
            x_rec = self.decode_to_img(z_out["coarse_content"], z_out["fine_content"], z_out["coarse_position"], z_out["fine_position"])

            log["inputs"] = x
            log["reconstructions"] = x_rec

        return log
    
    @torch.no_grad()
    def sample_from_scratch(
            self,
            c_coarse, c_fine, c_pos_coarse, c_pos_fine, c_seg_coarse, c_seg_fine, 
            temperature = 1.0, sample = True, top_k = None, top_p = None, top_k_pos = None, top_p_pos = None, process = True, fix_fine_position = False,
        ):
        if self.activate_sos_for_fine_sequence:
            x_coarse, x_fine, x_pos_coarse, x_pos_fine, x_seg_coarse, x_seg_fine = c_coarse, c_fine, c_pos_coarse, c_pos_fine, c_seg_coarse, c_seg_fine
        else:
            x_coarse, x_fine, x_pos_coarse, x_pos_fine, x_seg_coarse, x_seg_fine = c_coarse, c_fine[:, :0], c_pos_coarse, c_pos_fine[:, :0], c_seg_coarse, c_seg_fine[:, :0]
        batch_size, device = x_coarse.size(0), x_coarse.device

        sample_coarse_flag = torch.zeros(batch_size, 1).to(device)
        while not torch.all(sample_coarse_flag.bool()):  # sample coarse grain code, torch.all(sample_coarse_flag) is True only if all elements are True
            # sample position first
            position_hidden, position_logits = self.transformer.sample_coarse_position(coarse_content=x_coarse, coarse_position=x_pos_coarse, coarse_seg=x_seg_coarse)
            position_logits = position_logits[:, -1, :] / temperature
            position_logits = self.avoid_repeat_or_enforce_pad_for_coarse_position(position_logits, x_pos_coarse, sample_coarse_flag)
            if top_k_pos is not None:
                position_logits = top_k_logits(position_logits, top_k_pos)
            # apply softmax to convert to probabilities
            position_probs = F.softmax(position_logits, dim=-1)
            if top_p_pos is not None:
                position_probs = top_p_logits(position_probs, top_p_pos)
            # sample from the distribution or take the most likely
            if sample:
                ix_pos = torch.multinomial(position_probs, num_samples=1)
            else:
                _, ix_pos = torch.topk(position_probs, k=1, dim=-1)

            # append to the sequence and continue
            x_pos_coarse = torch.cat((x_pos_coarse, ix_pos), dim=1)

            # 如果某一个样本sample到了<eos> code，则该样本后续均为<pad> code
            sample_coarse_flag = sample_coarse_flag + (ix_pos == self.coarse_position_eos_code)

            # sample content next
            _, content_logits = self.transformer.sample_coarse_content(coarse_content=None, coarse_position=x_pos_coarse, coarse_seg=None, position_hidden=position_hidden)
            content_logits = content_logits[:, -1, :] / temperature
            content_logits = self.avoid_special_or_enforce_pad_for_content(content_logits, sample_coarse_flag)
            if top_k is not None:
                content_logits = top_k_logits(content_logits, top_k)
            # apply softmax to convert to probabilities
            content_probs = F.softmax(content_logits, dim=-1)
            if top_p is not None:
                content_probs = top_p_logits(content_probs, top_p)
            # sample from the distribution or take the most likely
            if sample:
                ix = torch.multinomial(content_probs, num_samples=1)
            else:
                _, ix = torch.topk(content_probs, k=1, dim=-1)
            
            if self.activate_segment:
                x_seg_coarse = torch.cat([x_seg_coarse, torch.zeros(batch_size, 1).long().to(device)], dim=1)
            else:
                pass

            # append to the sequence and continue
            x_coarse = torch.cat((x_coarse, ix), dim=1)

            if process:
                print("\r sampling coarse: {}".format(x_coarse.size(1)), end="")
        
        if not fix_fine_position: # sample with sampled fine position
            transfered_fine_position = self.transfer_sampled_coarse_position_to_sampled_fine_position(x_pos_coarse)
            sample_fine_flag = torch.zeros(batch_size, 1).to(device)

            while not torch.all(sample_fine_flag.bool()):  # sample coarse grain code, torch.all(sample_coarse_flag) is True only if all elements are True
                position_hidden, position_logits = self.transformer.sample_fine_position(coarse_content=x_coarse, fine_content=x_fine, coarse_position=x_pos_coarse, fine_position=x_pos_fine, coarse_seg=x_seg_coarse, fine_seg=x_seg_fine)                
                position_logits = position_logits[:, -1, :] / temperature
                position_logits = self.avoid_repeat_or_enforce_pad_for_fine_position(position_logits, transfered_fine_position, sample_fine_flag)
                if top_k_pos is not None:
                    position_logits = top_k_logits(position_logits, top_k_pos)
                # apply softmax to convert to probabilities
                position_probs = F.softmax(position_logits, dim=-1)
                if top_p_pos is not None:
                    position_probs = top_p_logits(position_probs, top_p_pos)
                # sample from the distribution or take the most likely
                if sample:
                    ix_pos = torch.multinomial(position_probs, num_samples=1)
                else:
                    _, ix_pos = torch.topk(position_probs, k=1, dim=-1)
                
                # append to the sequence and continue
                x_pos_fine = torch.cat((x_pos_fine, ix_pos), dim=1)
                transfered_fine_position = torch.cat([transfered_fine_position, ix_pos], dim=1)

                # 如果某一个样本sample到了<eos> code，则该样本后续均为<pad> code
                sample_fine_flag = sample_fine_flag + (ix_pos == self.fine_position_eos_code)

                # sample content next
                _, content_logits = self.transformer.sample_fine_content(coarse_content=x_coarse, fine_content=x_fine, coarse_position=x_pos_coarse, fine_position=x_pos_fine, coarse_seg=x_seg_coarse, fine_seg=x_seg_fine, position_hidden=position_hidden)
                content_logits = content_logits[:, -1, :] / temperature
                content_logits = self.avoid_special_or_enforce_pad_for_content(content_logits, sample_fine_flag)
                if top_k is not None:
                    content_logits = top_k_logits(content_logits, top_k)
                # apply softmax to convert to probabilities
                content_probs = F.softmax(content_logits, dim=-1)
                if top_p is not None:
                    content_probs = top_p_logits(content_probs, top_p)
                # sample from the distribution or take the most likely
                if sample:
                    ix = torch.multinomial(content_probs, num_samples=1)
                else:
                    _, ix = torch.topk(content_probs, k=1, dim=-1)

                # append to the sequence and continue
                x_fine = torch.cat((x_fine, ix), dim=1)

                if self.activate_segment:
                    x_seg_fine = torch.cat([x_seg_fine, torch.ones(batch_size, 1).long().to(device)], dim=1)
                else:
                    pass

                if process:
                    print("\r sampled coarse size: {} ; sampling fine: {}".format(x_coarse.size(1), x_fine.size(1)), end="")
        else:
            transfered_remain_fine_position = self.transfer_sampled_coarse_position_to_remain_fine_position(x_pos_coarse)
            sample_fine_flag = torch.zeros(batch_size, 1).to(device)

            for fine_index in trange(transfered_remain_fine_position.size(1)):
                if self.activate_sos_for_fine_sequence and fine_index == 0:
                    continue

                ix_pos = transfered_remain_fine_position[:, fine_index].unsqueeze(-1)
                
                # append to the sequence and continue
                x_pos_fine = torch.cat((x_pos_fine, ix_pos), dim=1)
                    
                sample_fine_flag = sample_fine_flag + (ix_pos == self.fine_position_eos_code)

                # sample content next
                _, content_logits = self.transformer.sample_fine_content(coarse_content=x_coarse, fine_content=x_fine, coarse_position=x_pos_coarse, fine_position=x_pos_fine, coarse_seg=x_seg_coarse, fine_seg=x_seg_fine, position_hidden=None)
                content_logits = content_logits[:, -1, :] / temperature
                content_logits = self.avoid_special_or_enforce_pad_for_content(content_logits, sample_fine_flag)
                if top_k is not None:
                    content_logits = top_k_logits(content_logits, top_k)
                # apply softmax to convert to probabilities
                content_probs = F.softmax(content_logits, dim=-1)
                if top_p is not None:
                    content_probs = top_p_logits(content_probs, top_p)
                # sample from the distribution or take the most likely
                if sample:
                    ix = torch.multinomial(content_probs, num_samples=1)
                else:
                    _, ix = torch.topk(content_probs, k=1, dim=-1)

                # append to the sequence and continue
                x_fine = torch.cat((x_fine, ix), dim=1)

                if self.activate_segment:
                    x_seg_fine = torch.cat([x_seg_fine, torch.ones(batch_size, 1).long().to(device)], dim=1)
                else:
                    pass

                if process:
                    print("\r sampled coarse size: {} ; sampling fine: {}".format(x_coarse.size(1), x_fine.size(1)), end="")

        # cut off conditioning
        x_coarse = x_coarse[:, c_coarse.shape[1]:]
        x_pos_coarse = x_pos_coarse[:, c_pos_coarse.shape[1]:]
        if self.activate_sos_for_fine_sequence:
            x_fine = x_fine[:, c_fine.shape[1]:]
            x_pos_fine = x_pos_fine[:,c_fine.shape[1]:]
        return x_coarse, x_fine, x_pos_coarse, x_pos_fine
    
    def transfer_sampled_coarse_position_to_remain_fine_position(self, coarse_position):
        c_position = self.fine_position_sos_code * torch.ones_like(coarse_position[:, :1]).long().to(coarse_position.device)
        coarse_position = coarse_position[:, 1:]
        batch_size, coarse_length = coarse_position.size()
        device = coarse_position.device
        grain_indices = torch.ones(batch_size, int(self.hw1 * self.hw1)).long()  ## NOTE
        for i in range(batch_size):
            for coarse_l in range(coarse_length):
                if coarse_position[i, coarse_l] == self.coarse_position_eos_code:
                    break
                else:
                    grain_indices[i, coarse_position[i, coarse_l]] = 0  # NOTE
        
        grain_indices = rearrange(grain_indices, "B (h w) -> B h w", h=self.hw1, w=self.hw1)
        # in fact here it is no need to distinguish region-first or row-first since we only need to know which positions are sampled
        if self.fine_position_order == "region-first":
            fine_position_list = [torch.cat([self.position_sequence_fine[grain_indices[i] == 1].view(-1).to(device), self.fine_position_eos_tensor.to(device)]) for i in range(batch_size)]
            fine_position_tensor = pad_sequence(fine_position_list, batch_first=True, padding_value=self.fine_position_pad_code)
        elif self.fine_position_order == "row-first":
            grain_indices = grain_indices.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
            fine_position_list = [torch.cat([self.position_sequence_fine[grain_indices[i] == 1].to(device), self.fine_position_eos_tensor.to(device)]) for i in range(batch_size)]
            fine_position_tensor = pad_sequence(fine_position_list, batch_first=True, padding_value=self.fine_position_pad_code)
        
        if self.activate_sos_for_fine_sequence:
            fine_position_tensor = torch.cat([c_position, fine_position_tensor], dim=1)
        return fine_position_tensor
    
    def transfer_sampled_coarse_position_to_sampled_fine_position(self, coarse_position):
        c_position = self.fine_position_sos_code * torch.ones_like(coarse_position[:, :1]).long().to(coarse_position.device)
        coarse_position = coarse_position[:, 1:]
        batch_size, coarse_length = coarse_position.size()
        device = coarse_position.device
        grain_indices = torch.zeros(batch_size, int(self.hw1 * self.hw1)).long()
        for i in range(batch_size):
            for coarse_l in range(coarse_length):
                if coarse_position[i, coarse_l] == self.coarse_position_eos_code:
                    break
                else:
                    grain_indices[i, coarse_position[i, coarse_l]] = 1  # means this coarse-position is sampled and will transfer to fine position later
        
        grain_indices = rearrange(grain_indices, "B (h w) -> B h w", h=self.hw1, w=self.hw1)
        # in fact here it is no need to distinguish region-first or row-first since we only need to know which positions are sampled
        if self.fine_position_order == "region-first":
            fine_position_list = [torch.cat([self.position_sequence_fine[grain_indices[i] == 1].view(-1).to(device), self.fine_position_eos_tensor.to(device)]) for i in range(batch_size)]
            fine_position_tensor = pad_sequence(fine_position_list, batch_first=True, padding_value=self.fine_position_pad_code)
        elif self.fine_position_order == "row-first":
            grain_indices = grain_indices.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
            fine_position_list = [torch.cat([self.position_sequence_fine[grain_indices[i] == 1].to(device), self.fine_position_eos_tensor.to(device)]) for i in range(batch_size)]
            fine_position_tensor = pad_sequence(fine_position_list, batch_first=True, padding_value=self.fine_position_pad_code)
        
        if self.activate_sos_for_fine_sequence:
            fine_position_tensor = torch.cat([c_position, fine_position_tensor], dim=1)
        return fine_position_tensor
    
    def avoid_repeat_or_enforce_pad_for_coarse_position(self, logits, sampled_position, flag):
        batch_size = logits.size(0)
        out = logits.clone()
        for i in range(batch_size):
            if flag[i] == 0:  # avoid sample <sos>, <pad>, finer position and sampled coarse position
                out[i, sampled_position[i]] = -float('Inf')  # forbid <sos> and sampled coarse position
                out[i, self.coarse_position_pad_code] = -float('Inf')  # forbid <pad>
                out[i, self.max_coarse_postion_idx:] = -float('Inf')  # forbid finer position
                out[i, self.coarse_position_eos_code] = logits[i, self.coarse_position_eos_code]
            else:  # enforce sample <pad>
                out[i, :] = -float('Inf')
                out[i, self.coarse_position_pad_code] = logits[i, self.coarse_position_pad_code]
        return out
    
    def avoid_repeat_or_enforce_pad_for_fine_position(self, logits, sampled_position, flag):
        batch_size = logits.size(0)
        out = logits.clone()
        for i in range(batch_size):
            if flag[i] == 0:  # avoid repeat and <sos>, <pad>
                out[i, sampled_position[i]] = -float('Inf')  # forbid <sos>
                out[i, self.fine_position_pad_code] = -float('Inf')  # forbid <pad>
                out[i, self.fine_position_eos_code] = logits[i, self.fine_position_eos_code]  # keep <eos>
                out[i, self.fine_position_sos_code] = -float('Inf')
            else:
                out[i, :] = -float('Inf')
                out[i, self.fine_position_pad_code] = logits[i, self.fine_position_pad_code]
        return out
    
    def avoid_special_or_enforce_pad_for_content(self, logits, flag):
        batch_size = logits.size(0)
        out = logits.clone()
        for i in range(batch_size):
            if flag[i] == 0:  # avoid special
                out[i, self.content_pad_code] = -float('Inf')
                out[i, self.content_eos_code:] = -float('Inf')  # avoid class-label sampling
                # out[i, self.content_sos_code] = -float('Inf')
            else:  # enfore pad
                out[i, :] = -float('Inf')
                out[i, self.content_pad_code] = logits[i, self.content_pad_code]
        return out