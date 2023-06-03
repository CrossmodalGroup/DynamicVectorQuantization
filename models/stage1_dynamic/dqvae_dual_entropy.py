import torch
from torch import nn, Tensor
import pytorch_lightning as pl
from functools import partial
from einops import rearrange
from utils.utils import instantiate_from_config
import torch.nn.functional as F

from modules.dynamic_modules.utils import draw_dual_grain_256res, draw_dual_grain_256res_color
from models.stage1.utils import Scheduler_LinearWarmup, Scheduler_LinearWarmup_CosineDecay
from models.stage2.utils import disabled_train

class Entropy(nn.Sequential):
    def __init__(self, patch_size, image_width, image_height):
        super(Entropy, self).__init__()
        self.width = image_width
        self.height = image_height
        self.psize = patch_size
        # number of patches per image
        self.patch_num = int(self.width * self.height / self.psize ** 2)
        self.hw = int(self.width // self.psize)
        # unfolding image to non overlapping patches
        self.unfold = torch.nn.Unfold(kernel_size=(self.psize, self.psize), stride=self.psize)

    def entropy(self, values: torch.Tensor, bins: torch.Tensor, sigma: torch.Tensor, batch: int) -> torch.Tensor:
        """Function that calculates the entropy using marginal probability distribution function of the input tensor
            based on the number of histogram bins.
        Args:
            values: shape [BxNx1].
            bins: shape [NUM_BINS].
            sigma: shape [1], gaussian smoothing factor.
            batch: int, size of the batch
        Returns:
            torch.Tensor:
        """
        epsilon = 1e-40
        values = values.unsqueeze(2)
        residuals = values - bins.unsqueeze(0).unsqueeze(0)
        kernel_values = torch.exp(-0.5 * (residuals / sigma).pow(2))

        pdf = torch.mean(kernel_values, dim=1)
        normalization = torch.sum(pdf, dim=1).unsqueeze(1) + epsilon
        pdf = pdf / normalization + epsilon
        entropy = - torch.sum(pdf * torch.log(pdf), dim=1)
        entropy = entropy.reshape((batch, -1))
        entropy = rearrange(entropy, "B (H W) -> B H W", H=self.hw, W=self.hw)
        return entropy

    def forward(self, inputs: Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        gray_images = 0.2989 * inputs[:, 0:1, :, :] + 0.5870 * inputs[:, 1:2, :, :] + 0.1140 * inputs[:, 2:, :, :]

        # create patches of size (batch x patch_size*patch_size x h*w/ (patch_size*patch_size))
        unfolded_images = self.unfold(gray_images)
        # reshape to (batch * h*w/ (patch_size*patch_size) x (patch_size*patch_size)
        unfolded_images = unfolded_images.transpose(1, 2)
        unfolded_images = torch.reshape(unfolded_images.unsqueeze(2),
                                        (unfolded_images.shape[0] * self.patch_num, unfolded_images.shape[2]))

        entropy = self.entropy(unfolded_images, bins=torch.linspace(0, 1, 32).to(device=inputs.device),
                               sigma=torch.tensor(0.01), batch=batch_size)

        return entropy

class DualGrainVQModel(pl.LightningModule):
    def __init__(self,
                 encoderconfig,
                 decoderconfig,
                 lossconfig,
                 vqconfig,

                 quant_before_dim,
                 quant_after_dim,
                 quant_sample_temperature = 0., 
                 ckpt_path = None,
                 ignore_keys = [],
                 image_key = "image",
                 monitor = None,
                 warmup_epochs = 0,
                 loss_with_epoch = True,
                 scheduler_type = "linear-warmup_cosine-decay",
                 entropy_patch_size = 16, # maximum patch size of all granularity
                 image_size = 256,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = instantiate_from_config(encoderconfig)
        self.decoder = instantiate_from_config(decoderconfig)
        self.loss = instantiate_from_config(lossconfig)

        self.quantize = instantiate_from_config(vqconfig)

        self.quant_conv = torch.nn.Conv2d(quant_before_dim, quant_after_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(quant_after_dim, quant_before_dim, 1)
        self.quant_sample_temperature = quant_sample_temperature

        self.entropy_patch_size = entropy_patch_size
        self.image_size = image_size 
        self.entropy_calculation = Entropy(entropy_patch_size, image_size, image_size)
        self.entropy_calculation = self.entropy_calculation.eval()
        self.entropy_calculation.train = disabled_train

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if monitor is not None:
            self.monitor = monitor

        self.warmup_epochs = warmup_epochs
        self.loss_with_epoch = loss_with_epoch
        self.scheduler_type = scheduler_type

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        x_entropy = self.entropy_calculation(x)
        h_dict = self.encoder(x, x_entropy)
        h = h_dict["h_dual"]
        grain_indices = h_dict["indices"]
        codebook_mask = h_dict["codebook_mask"]
        gate = h_dict["gate"]

        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(x=h, temp=self.quant_sample_temperature, codebook_mask=codebook_mask)
        return quant, emb_loss, info, grain_indices, gate, x_entropy

    def decode(self, quant, grain_indices=None):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant, grain_indices)
        return dec

    def forward(self, input):
        quant, diff, _, grain_indices, gate, x_entropy = self.encode(input)
        dec = self.decode(quant, grain_indices)
        return dec, diff, grain_indices, gate, x_entropy

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        if x.size(1) != 3:
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss, indices, gate, x_entropy = self(x)
        ratio = indices.sum() / (indices.size(0) * indices.size(1) * indices.size(2))

        if optimizer_idx == 0:
            # autoencode
            if self.loss_with_epoch:
                aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.current_epoch, last_layer=self.get_last_layer(), split="train", gate=gate)
            else:
                aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step, last_layer=self.get_last_layer(), split="train", gate=gate)
            
            self.log("train_aeloss", aeloss, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            self.log("train_fine_ratio", ratio, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            rec_loss = log_dict_ae["train_rec_loss"]
            self.log("train_rec_loss", rec_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            del log_dict_ae["train_rec_loss"]
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            if self.loss_with_epoch:
                discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.current_epoch, last_layer=self.get_last_layer(), split="train")
            else:
                discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step, last_layer=self.get_last_layer(), split="train")

            self.log("train_discloss", discloss, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss, indices, gate, x_entropy = self(x)
        ratio = indices.sum() / (indices.size(0) * indices.size(1) * indices.size(2))
        self.log("val_fine_ratio", ratio, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        if self.loss_with_epoch:
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.current_epoch, last_layer=self.get_last_layer(), split="val", gate=gate)
            discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.current_epoch, last_layer=self.get_last_layer(), split="val")
        else:
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step, last_layer=self.get_last_layer(), split="val", gate=gate)
            discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step, last_layer=self.get_last_layer(), split="val")

        rec_loss = log_dict_ae["val_rec_loss"]
        self.log("val_rec_loss", rec_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        del log_dict_ae["val_rec_loss"]
        self.log("val_aeloss", aeloss, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return self.log_dict
    
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
        
        warmup_steps = self.steps_per_epoch * self.warmup_epochs

        if self.scheduler_type == "linear-warmup":
            scheduler_ae = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(opt_ae, Scheduler_LinearWarmup(warmup_steps)), "interval": "step", "frequency": 1,
            }
            scheduler_disc = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(opt_disc, Scheduler_LinearWarmup(warmup_steps)), "interval": "step", "frequency": 1,
            }
        elif self.scheduler_type == "linear-warmup_cosine-decay":
            multipler_min = self.min_learning_rate / self.learning_rate
            scheduler_ae = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(opt_ae, Scheduler_LinearWarmup_CosineDecay(warmup_steps=warmup_steps, max_steps=self.training_steps, multipler_min=multipler_min)), "interval": "step", "frequency": 1,
            }
            scheduler_disc = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(opt_disc, Scheduler_LinearWarmup_CosineDecay(warmup_steps=warmup_steps, max_steps=self.training_steps, multipler_min=multipler_min)), "interval": "step", "frequency": 1,
            }
        else:
            raise NotImplementedError()

        return [opt_ae, opt_disc], [scheduler_ae, scheduler_disc]

    def get_last_layer(self):
        try:
            return self.decoder.conv_out.weight
        except:
            return self.decoder.last_layer

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _, grain_indices, gate, x_entropy = self(x)

        log["inputs"] = x
        log["reconstructions"] = xrec
        # log["grain"] = draw_dual_grain_256res(images=x.clone(), indices=grain_indices)
        log["grain_map"] = draw_dual_grain_256res_color(images=x.clone(), indices=grain_indices, scaler=0.7)
        x_entropy = x_entropy.sub(x_entropy.min()).div(max(x_entropy.max() - x_entropy.min(), 1e-5))
        log["entropy_map"] = draw_dual_grain_256res_color(images=x.clone(), indices=x_entropy, scaler=0.7)
        return log
    
    def get_code_emb_with_depth(self, code):
        embed = self.quantize.get_codebook_entry(code)
        # embed = rearrange(embed, "b h w c -> b c h w")
        return embed
        # return self.quantize.embed_code_with_depth(code)