# for pytorch_lightning ModelCheckpoint, Callback, LearningRateMonitor, ... modules
import os
import wandb
from omegaconf import OmegaConf
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
import torch
import torchvision
from PIL import Image

class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config, argv_content=None):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config
    
        self.argv_content = argv_content

    # 在pretrain例程开始时调用。
    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))
            
            with open(os.path.join(self.logdir, "argv_content.txt"), "w") as f:
                f.write(str(self.argv_content))
        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass

class CaptionImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp, type="wandb"):
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.clamp = clamp
        self.logger_log_images = {
            pl.loggers.WandbLogger: self._wandb,
            pl.loggers.TensorBoardLogger: self._tensorboard,
        }
        self.type = type  # wandb or tensorboard


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *kwargs):
        self.log_img(pl_module, batch, batch_idx, split="train")
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *kwargs):
        self.log_img(pl_module, batch, batch_idx, split="val")
    
    @rank_zero_only
    def _wandb(self, pl_module, images, batch_idx, split):
        grids = dict()
        for k in images:
            grid = torchvision.utils.make_grid(images[k], normalize=True)
            grids[f"{split}/{k}"] = wandb.Image(grid)
        pl_module.logger.experiment.log(grids, commit=False)
    
    @rank_zero_only
    def _tensorboard(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4, normalize=True)
            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(tag, grid, global_step=pl_module.global_step)
    
    @rank_zero_only
    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if (batch_idx % self.batch_freq == 0) and hasattr(pl_module, "log_images") and callable(pl_module.log_images) and (self.max_images > 0):
            logger = type(pl_module.logger)
            is_train = pl_module.training
            if is_train:
                pl_module.eval()
            with torch.no_grad():
                images = pl_module.log_images(batch, split=split)
            
            # NOTE: 集群的路径总是有bug！！！！！！！！
            if "groundtruth_captions" in images:
                # if self.type == "wandb":
                #     pl_module.logger.log_text(key="samples_{}".format(pl_module.global_step), columns=["{}_groundtruth_captions".format(split)], data=images['groundtruth_captions'])
                # else:
                #     pl_module.logger.experiment.add_text("{}_groundtruth_captions".format(split), str(images['groundtruth_captions']), global_step=pl_module.global_step)
                del images['groundtruth_captions']
            
            if "dest_captions" in images:
                # if self.type == "wandb":
                #     pl_module.logger.log_text(key="samples_{}".format(pl_module.global_step), columns=["{}_dest_captions".format(split)], data=images['dest_captions'])
                # else:
                #     pl_module.logger.experiment.add_text("{}_dest_captions".format(split), str(images['dest_captions']), global_step=pl_module.global_step)
                del images['dest_captions']
            
            if "sample_captions" in images:
                # if self.type == "wandb":
                #     pl_module.logger.log_text(key="samples_{}".format(pl_module.global_step), columns=["{}_sample_captions".format(split)], data=images['sample_captions'])
                # else:
                #     pl_module.logger.experiment.add_text("{}_sample_captions".format(split), str(images['sample_captions']), global_step=pl_module.global_step)
                del images['sample_captions']

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)
            self.log_local(pl_module.logger.save_dir, split, images, pl_module.global_step, pl_module.current_epoch, batch_idx)
            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)
            
            if is_train:
                pl_module.train()
    
    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4, normalize=True)
            grid = grid.transpose(0,1).transpose(1,2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid*255).astype(np.uint8)
            filename = "Step_{:06}-Epoch_{:03}-Batch_{:06}-{}.png".format(global_step,current_epoch,batch_idx,k)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)


if __name__ == "__main__":
    pass