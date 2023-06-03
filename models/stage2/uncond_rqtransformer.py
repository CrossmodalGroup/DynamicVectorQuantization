import math
import os
import sys
from functools import partial
from einops import rearrange
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

sys.path.append(os.getcwd())

from utils.utils import (
    SOSProvider, Labelator,
    instantiate_from_config
)
from models.stage2.utils import learning_rate_schedule, disabled_train

class RQTransformerTrainer(pl.LightningModule):
    def __init__(self, 
                 transformer_config,
                 first_stage_config,
                 ckpt_path=None,
                 ignore_keys=[],
                 monitor=None,
                 weight_decay=0.01,
                 warmup_epochs=0,
                 ):
        super().__init__()
        self.first_stage_key = "image"
        self.cond_stage_key = "image"
        self.init_first_stage_from_ckpt(first_stage_config)
        self.transformer = instantiate_from_config(config=transformer_config)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
    
        print(f"Using no cond stage. Assuming the training is intended to be unconditional. "
            f"Prepending 0 as a sos token.")
        self.cond_stage_model = SOSProvider(0)
    
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs

        if monitor is not None:
            self.monitor = monitor
    
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
        no_decay.add('pos_emb_cond')
        no_decay.add('pos_emb_hw')
        no_decay.add('pos_emb_d')

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
    
    def training_step(self, batch, batch_idx):
        x, c = self.get_xc(batch)
        loss = self(x, c)
        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, c = self.get_xc(batch)
        loss = self(x, c)
        self.log("val_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss
    
    def forward(self, x, c):
        z_indices = self.encode_to_z(x)
        _, c_indices = self.encode_to_c(c)
        loss = self.transformer(idx=z_indices, c_idx=c_indices, model_aux=self.first_stage_model, return_loss=True)
        return loss
    
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
        code = self.first_stage_model.get_codes(x)
        return code

    @torch.no_grad()
    def decode_to_img(self, code):
        x = self.first_stage_model.decode_code(code)
        return x
    
    @torch.no_grad()
    def encode_to_c(self, c):
        quant_c, _, [_,_,indices] = self.cond_stage_model.encode(c)
        if len(indices.shape) > 2:
            indices = indices.view(c.shape[0], -1)
        return quant_c, indices

    @torch.no_grad()
    def log_images(self, batch, temperature=1.0, top_k=300, top_p=0.95, **kwargs):
        log = dict()

        N = 16
        x, c = self.get_xc(batch, N)
        x = x.to(device=self.device)
        c = c.to(device=self.device)

        z_indices = self.encode_to_z(x)
        quant_c, c_indices = self.encode_to_c(c)

        # reconstruction
        x_rec = self.decode_to_img(z_indices)

        log["inputs"] = x
        log["reconstructions"] = x_rec

        # sample 
        sample_code = self.transformer.sample(
            partial_sample=torch.randint(0, 1, (min(N,c_indices.size(0)), 8, 8, 4)).to(self.device),
            model_aux=self.first_stage_model,
            cond=c_indices,
            start_loc=(0, 0),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            is_tqdm=True,
            desc="Sampling",
            fast=True,
        )
        log["sample"] = self.decode_to_img(sample_code)

        return log



if __name__ == "__main__":
    pass