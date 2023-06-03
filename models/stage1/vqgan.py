import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from functools import partial
from utils.utils import instantiate_from_config

from modules.diffusionmodules.model import Encoder, Decoder
from modules.vector_quantization.quantize_vqgan import VectorQuantizer2 as VectorQuantizer

# linear warmup functions, Test done: Works fine
def fn(warmup_steps, step):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    else:
        return 1.0

def linear_warmup_decay(warmup_steps):
    return partial(fn, warmup_steps)

class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path = None,
                 ignore_keys = [],
                 image_key = "image",
                 colorize_nlabels = None,
                 monitor = None,
                 remap = None,
                 warmup_epochs = 0,
                 loss_with_epoch = True, 
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, legacy=False)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        
        # for warm up
        self.warmup_epochs = warmup_epochs
        self.loss_with_epoch = loss_with_epoch  # determine whether the loss is calculated by self.global_step or self.current_epoch

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
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info
    
    def encode2(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info, h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        if x.size(1) != 3:
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)

        if optimizer_idx == 0:
            # autoencode
            if self.loss_with_epoch:
                aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.current_epoch,
                                                last_layer=self.get_last_layer(), split="train")
            else:
                aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("train_aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            rec_loss = log_dict_ae["train_rec_loss"]
            self.log("train_rec_loss", rec_loss,
                    prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            del log_dict_ae["train_rec_loss"]
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            if self.loss_with_epoch:
                discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.current_epoch,
                                                last_layer=self.get_last_layer(), split="train")
            else:
                discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
            self.log("train_discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        if self.loss_with_epoch:
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.current_epoch,
                                                last_layer=self.get_last_layer(), split="val")

            discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.current_epoch,
                                                last_layer=self.get_last_layer(), split="val")
        else:
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                                last_layer=self.get_last_layer(), split="val")

            discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                                last_layer=self.get_last_layer(), split="val")

        rec_loss = log_dict_ae["val_rec_loss"]
        self.log("val_rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        del log_dict_ae["val_rec_loss"]
        self.log("val_aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict
    
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        
        warmup_steps = self.steps_per_epoch * self.warmup_epochs

        scheduler_ae = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                opt_ae,
                linear_warmup_decay(warmup_steps),
            ),
            "interval": "step",
            "frequency": 1,
        }
        scheduler_disc = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                opt_disc,
                linear_warmup_decay(warmup_steps),
            ),
            "interval": "step",
            "frequency": 1,
        }
        
        return [opt_ae, opt_disc], [scheduler_ae, scheduler_disc]

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x
    
    def get_code_emb_with_depth(self, code):
        return self.quantize.embed_code_with_depth(code)