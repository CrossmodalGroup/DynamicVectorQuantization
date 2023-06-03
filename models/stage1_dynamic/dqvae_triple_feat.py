import torch
import pytorch_lightning as pl
from functools import partial
from utils.utils import instantiate_from_config
from modules.dynamic_modules.utils import draw_triple_grain_256res, draw_triple_grain_256res_color
from models.stage1.utils import Scheduler_LinearWarmup, Scheduler_LinearWarmup_CosineDecay

def linear_warmup(warmup_steps):
    def linear_warmup_fn(warmup_steps, step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            return 1.0
    return partial(linear_warmup_fn, warmup_steps)


class TripleGrainVQModel(pl.LightningModule):
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


        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        if monitor is not None:
            self.monitor = monitor
        
        # for learning rate scheduler
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
        h_dict = self.encoder(x, None)
        h = h_dict["h_triple"]
        grain_indices = h_dict["indices"]
        codebook_mask = h_dict["codebook_mask"]
        gate = h_dict["gate"]

        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(x=h, temp=self.quant_sample_temperature, codebook_mask=codebook_mask)
        return quant, emb_loss, info, grain_indices, gate

    def decode(self, quant, grain_indices=None):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant, grain_indices=None)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _, grain_indices, gate = self.encode(input)
        dec = self.decode(quant, grain_indices)
        return dec, diff, grain_indices, gate

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        if x.size(1) != 3:
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss, indices, gate = self(x)

        indices_sum = indices.size(0) * indices.size(1) * indices.size(2)
        fine_radio = (indices==2).sum() / indices_sum
        median_radio = (indices==1).sum() / indices_sum

        if optimizer_idx == 0:
            # autoencode
            if self.loss_with_epoch:
                aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.current_epoch, last_layer=self.get_last_layer(), split="train", gate=gate)
            else:
                aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step, last_layer=self.get_last_layer(), split="train", gate=gate)

            self.log("train_aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

            self.log("train_fine_radio", fine_radio, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train_median_radio", median_radio, prog_bar=True, logger=True, on_step=True, on_epoch=True)

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
            self.log("train_discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss, indices, gate = self(x)
        
        indices_sum = indices.size(0) * indices.size(1) * indices.size(2)
        fine_radio = (indices==2).sum() / indices_sum
        median_radio = (indices==1).sum() / indices_sum

        self.log("val_fine_radio", fine_radio, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("val_fine_radio", median_radio, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        if self.loss_with_epoch:
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.current_epoch, last_layer=self.get_last_layer(), split="val", gate=gate)
            discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.current_epoch, last_layer=self.get_last_layer(), split="val")
        else:
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step, last_layer=self.get_last_layer(), split="val", gate=gate)
            discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step, last_layer=self.get_last_layer(), split="val")

        rec_loss = log_dict_ae["val_rec_loss"]
        self.log("val_rec_loss", rec_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        del log_dict_ae["val_rec_loss"]
        self.log("val_aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
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
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _, grain_indices, gate = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        log["grain"] = draw_triple_grain_256res(images=x.clone(), indices=grain_indices)
        log["grain_color"] = draw_triple_grain_256res_color(images=x.clone(), indices=grain_indices)
        return log
    
    def get_code_emb_with_depth(self, code):
        return self.quantize.embed_code_with_depth(code)


if __name__ == "__main__":
    pass