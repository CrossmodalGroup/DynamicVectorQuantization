# NOTE: 相比较于1, 增加了segmenation embedding 的输入

import math
import os
import sys
from functools import partial

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

sys.path.append(os.getcwd())

from utils.utils import instantiate_from_config
from models.stage2.utils import learning_rate_schedule, disabled_train

class Text2ImageTransformer(pl.LightningModule):
    def __init__(self, 
                 transformer_config,
                 first_stage_config,
                 permuter_config=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 first_stage_key="image",
                 cond_stage_key="caption",
                 
                 pkeep=1.0,
                 monitor=None,
                 
                 weight_decay=0.01,
                 warmup_epochs=0,
                 
                 text_loss_radio=0,
                 
                 text_vocab_size=0,
                 image_vocab_size=0,
                 ):
        super().__init__()
        print(f"Using text as condition.")
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.init_first_stage_from_ckpt(first_stage_config)
        if permuter_config is None:
            permuter_config = {"target": "modules.transformer.permuter.Identity"}
            
        self.permuter = instantiate_from_config(config=permuter_config)
        self.transformer = instantiate_from_config(config=transformer_config)
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        if monitor is not None:
            self.monitor = monitor
        self.pkeep = pkeep
        
        # new hype-parameter
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        
        self.calculate_text_loss = False
        if text_loss_radio > 0:
            self.calculate_text_loss = True
            self.text_loss_radio = text_loss_radio
            self.image_loss_radio = 1 - text_loss_radio
        self.text_vocab_size = text_vocab_size
        self.image_vocab_size = image_vocab_size
    
    def configure_optimizers(self):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
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
    
    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        if x.size(1) != 3 and len(x.shape) == 4:
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x
    
    def get_xc(self, batch, N=None):
        x = self.get_input(batch, self.first_stage_key)
        c = self.get_input(batch, self.cond_stage_key)
        if N is not None:
            x = x[:N]
            c = c[:N]
        return x, c
    
    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, _, info = self.first_stage_model.encode(x)
        indices = info[2].view(quant_z.shape[0], -1)
        indices = self.permuter(indices)
        return quant_z, indices

    @torch.no_grad()
    def decode_to_img(self, index, zshape):
        index = self.permuter(index, reverse=True)
        bhwc = (zshape[0],zshape[2],zshape[3],zshape[1])
        quant_z = self.first_stage_model.quantize.get_codebook_entry(
            index.reshape(-1), shape=bhwc)
        x = self.first_stage_model.decode(quant_z)
        return x
    
    def forward(self, x, c):
        # one step to produce the logits
        _, z_indices = self.encode_to_z(x)
        z_indices += self.text_vocab_size # [text_vocab_size, text_vocab_size + image_vocab_size]
        c_indices = c
        
        if self.training and self.pkeep < 1.0:
            mask = torch.bernoulli(self.pkeep*torch.ones(z_indices.shape,
                                                         device=z_indices.device))
            mask = mask.round().to(dtype=torch.int64)
            r_indices = torch.randint_like(z_indices, self.transformer.config.vocab_size)
            a_indices = mask*z_indices+(1-mask)*r_indices
        else:
            a_indices = z_indices
        
        batch_size, c_length, z_length = c_indices.size(0), c_indices.size(1), a_indices.size(1)
        seg_indices = torch.cat((
            torch.zeros(batch_size, c_length), torch.ones(batch_size, z_length-1)
            ), dim=1).long().to(self.device)
        cz_indices = torch.cat((c_indices, a_indices), dim=1)
        # print("c_indices.size(), a_indices.size(), ", c_indices.size(), a_indices.size())
        
        if self.calculate_text_loss:  # calculate both text and image modeling loss
            text_target = c_indices[:, 1:]
            image_target = z_indices
            
            logits, _ = self.transformer(idx=cz_indices[:, :-1], segment=seg_indices)
            
            logits_text = logits[:, :c_indices.shape[1]-1]
            logits_image = logits[:, c_indices.shape[1]-1:]
            # print("logits_text, logits_image:", logits_image.size(), logits_image.size())
            
            text_loss = F.cross_entropy(logits_text.reshape(-1, logits_text.size(-1)), text_target.reshape(-1))
            image_loss = F.cross_entropy(logits_image.reshape(-1, logits_image.size(-1)), image_target.reshape(-1))
            
            loss = self.text_loss_radio * text_loss + self.image_loss_radio * image_loss
            
            return loss, image_loss, text_loss
        else:  # calculate only image modeling loss
            image_target = z_indices
            logits, _ = self.transformer(idx=cz_indices[:, :-1], segment=seg_indices)
            logits_image = logits[:, c_indices.shape[1]-1:]
            # print("image_target, logits_image", image_target.size(), logits_image.size())
            image_loss = F.cross_entropy(logits_image.reshape(-1, logits_image.size(-1)), image_target.reshape(-1))
            
            return image_loss, image_loss, 0.
    
    def training_step(self, batch, batch_idx):
        x, c = self.get_xc(batch)
        loss, image_loss, text_loss = self(x, c)
        
        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train_image_loss", image_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train_text_loss", text_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, c = self.get_xc(batch)
        loss, image_loss, text_loss = self(x, c)
        self.log("val_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("val_image_loss", image_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("val_text_loss", text_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    @torch.no_grad()
    def log_images(self, batch, temperature=None, top_k=None, top_p=None, callback=None, lr_interface=False, **kwargs):
        log = dict()

        N = 4
        if lr_interface:
            x, c = self.get_xc(batch, N, diffuse=False, upsample_factor=8)
        else:
            x, c = self.get_xc(batch, N)
        x = x.to(device=self.device)
        c_indices = c.to(device=self.device)
        
        quant_z, z_indices = self.encode_to_z(x)
        z_indices += self.text_vocab_size # [text_vocab_size, text_vocab_size + image_vocab_size]
        
        # log["groundtruth_captions"] = self.tokenizer.decode_batch(c_indices.cpu().numpy().tolist())
        # log["groundtruth_captions"] = [[str(log["groundtruth_captions"])]]
        
        # sample
        z_start_indices = z_indices[:, :0]
        index_sample = self.sample(z_start_indices, c_indices,
                                   steps=z_indices.shape[1],
                                   temperature=temperature if temperature is not None else 1.0,
                                   sample=True,
                                   top_k=top_k if top_k is not None else 100,
                                   top_p=top_p if top_p is not None else 1.0, 
                                   callback=callback if callback is not None else lambda k: None)
        x_sample_nopix = self.decode_to_img(index_sample, quant_z.shape)
        
        # det sample
        z_start_indices = z_indices[:, :0]
        index_sample = self.sample(z_start_indices, c_indices,
                                   steps=z_indices.shape[1],
                                   sample=False,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample_det = self.decode_to_img(index_sample, quant_z.shape)

        # reconstruction
        z_indices -= self.text_vocab_size
        x_rec = self.decode_to_img(z_indices, quant_z.shape)
        
        if self.current_epoch == 0:
            log["inputs"] = x
            log["reconstructions"] = x_rec

        log["samples_nopix"] = x_sample_nopix
        log["samples_det"] = x_sample_det
        return log

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    def top_p_logits(self, probs, p):    
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        
        sorted_idx_remove_cond = cum_probs >= p
        
        sorted_idx_remove_cond[..., 1:] = sorted_idx_remove_cond[..., :-1].clone()
        sorted_idx_remove_cond[..., 0] = 0
        
        indices_to_remove = sorted_idx_remove_cond.scatter(-1, sorted_indices, sorted_idx_remove_cond)
        probs = probs.masked_fill(indices_to_remove, 0.0)
        norm_probs = probs / torch.sum(probs, dim=-1, keepdim=True)
        return norm_probs
    
    # avoid sample text token
    def avoid_text_sampling(self, logits):
        batch_size = logits.size(0)
        out = logits.clone()
        for i in range(batch_size):
            out[i, :self.text_vocab_size] = -float('Inf')
        return out
    
    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, sample=False, top_k=None, top_p=None,
               callback=lambda k: None):
        # NOTE: top_k -> softmax -> top_p
        # temperature (float): softmax temperature
        # top_k (Optional[int]): if given, sample using `top_k` logits
        # top_p (Optional[float]): if given, sample using `top_p` logits
        
        # assume using top_k sample or top_p sample ONLY
        # assert (top_k is None) or (top_p is None)
        
        x = torch.cat((c,x),dim=1)
        seg = torch.zeros(x.size(0), x.size(1)).long().to(x.device)
        block_size = self.transformer.get_block_size()
        assert not self.transformer.training
        
        for k in range(steps):
            callback(k)
            assert x.size(1) <= block_size # make sure model can see conditioning
            x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
            logits, _ = self.transformer(idx=x_cond, segment=seg)
            # pluck the logits at the final step and scale by temperature
            logits = logits[:, -1, :] / temperature

            # avoid sample text token
            logits = self.avoid_text_sampling(logits)

            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
                
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # optionally crop probabilities to only the top p prob
            if top_p is not None:
                probs = self.top_p_logits(probs, top_p)
                
            # sample from the distribution or take the most likely
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # append to the sequence and continue
            x = torch.cat((x, ix), dim=1)
            seg = torch.cat([seg, torch.ones(x.size(0), 1).long().to(x.device)], dim=1)
        # cut off conditioning
        x = x[:, c.shape[1]:]
        x -= self.text_vocab_size # remap to [0, image_vocab_size]
        return x