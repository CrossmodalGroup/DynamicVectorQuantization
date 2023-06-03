import datetime
import os
import sys

sys.path.append(os.getcwd())

import warnings

warnings.filterwarnings("ignore")

import argparse
import glob

# pytorch_lightning
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.trainer import Trainer
from utils.utils import instantiate_from_config

import pytz
Shanghai = pytz.timezone("Asia/Shanghai")
now = datetime.datetime.now().astimezone(Shanghai).strftime("%m-%dT%H-%M-%S")

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-b", "--base", nargs="*", metavar="base_config.yaml", help="paths to base configs. Loaded from left-to-right. Parameters can be overwritten or added with command-line options of the form `--key value`.", default=[],)
    parser.add_argument("-r", "--resume", type=str, const=True, default="", nargs="?", help="resume from logdir or checkpoint in logdir",)
    parser.add_argument("-s", "--seed", type=int, default=2021, help="seed for seed_everything",)
    parser.add_argument("-f", "--postfix", type=str, default="", help="post-postfix for default name",)
    parser.add_argument("-n", "--name", type=str, const=True, default="", nargs="?", help="postfix for logdir",)
    parser.add_argument("-l", "--logtype", type=str, default="wandb", nargs="?", help="log type", choices=["wandb","tensorboard"])
    parser.add_argument("-d", "--debug", type=str2bool, nargs="?", const=True, default=False, help="enable post-mortem debugging",)
    parser.add_argument("--activate_ddp_share", default=False, action="store_true",)
    parser.add_argument("-p", "--project", help="name of new or path to existing project", default="DynamicVectorQuantization")
    parser.add_argument("--save_n", default=3, type=int, help="save top-n with monitor or save every n epochs without monitor")

    return parser

def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))

if __name__ == "__main__":
    # now = datetime.datetime.now().strftime("%m-%dT%H-%M-%S")
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    opt, unknown = parser.parse_known_args()
    print("Current Workspace: ", str(os.getcwd()))
    print("Using Configs: {}".format(opt.base))

    # resume from checkpoint or logdir
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:  # resume from checkpoint
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            idx = len(paths)-paths[::-1].index("logs")+1
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
        else:  # resume from logdir
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs+opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[_tmp.index("logs")+1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        if opt.postfix != "":
            nowname = now + name + "_" + opt.postfix
        else:
            nowname = now + name
        logdir = os.path.join("logs", nowname)
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    # init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    
    trainer_config["gpus"] = opt.gpus
    trainer_config["precision"] = opt.precision
    for k in nondefault_trainer_args(opt):
        trainer_config[k] = getattr(opt, k)
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    # model
    model = instantiate_from_config(config.model)

    # trainer and callbacks
    trainer_kwargs = dict()
    os.makedirs(os.path.join(os.getcwd(),logdir), exist_ok=True)
    default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "project": opt.project,
                    "name": nowname,
                    "save_dir": str(os.path.join(os.getcwd(),logdir)),
                    "offline": opt.debug,
                    "id": nowname,
                }
            },
            "tensorboard": {
                "target": "pytorch_lightning.loggers.TensorBoardLogger",
                "params": {
                    "name": "tensorboard",
                    "save_dir": logdir,
                }
            },
        }
    default_logger_cfg = default_logger_cfgs[opt.logtype]
    logger_cfg = OmegaConf.create()
    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

    # model callback, reference: https://pytorch-lightning.readthedocs.io/en/latest/extensions/generated/pytorch_lightning.callbacks.ModelCheckpoint.html?highlight=callbacks.ModelCheckpoint#pytorch_lightning.callbacks.ModelCheckpoint
    if hasattr(model, "monitor"):
        filename = "{epoch}-{" + str(model.monitor) + ":.4f}"
        default_modelckpt_cfg = {
            "default_modelckpt_cfg": {
                "target": "pytorch_lightning.callbacks.ModelCheckpoint",
                "params": {
                    "dirpath": ckptdir,
                    "filename": filename,
                    "verbose": True,
                    "monitor": model.monitor,
                    "save_top_k": opt.save_n,
                    "every_n_epochs": opt.check_val_every_n_epoch,
                    "save_last": True,
                }
            }
        }
        print(f"Monitoring {model.monitor} as checkpoint metric.")
    else:
        default_modelckpt_cfg = {
            "default_modelckpt_cfg": {
                "target": "pytorch_lightning.callbacks.ModelCheckpoint",
                "params": {
                    "dirpath": ckptdir,
                    "filename": "{epoch}-{train_loss:.4f}-{val_loss:.4f}",
                    "verbose": True,
                    "every_n_epochs": int(opt.check_val_every_n_epoch),
                    "save_last": True,
                    "save_top_k": -1,
                }
            }
        }
    modelckpt_cfg = OmegaConf.create()
    modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)


    # add callback which sets up log directory
    default_callbacks_cfg = {
        "setup_callback": {
            "target": "utils.logger.SetupCallback",
            "params": {
                "resume": opt.resume,
                "now": now,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "config": config,
                "lightning_config": lightning_config,
                "argv_content": sys.argv + ["gpus: {}".format(torch.cuda.device_count())],
            }
        },
        # "richsummary_callback": {
        #     "target": "pytorch_lightning.callbacks.RichModelSummary",
        # },
        # reference: https://pytorch-lightning.readthedocs.io/en/latest/extensions/generated/pytorch_lightning.callbacks.RichModelSummary.html#pytorch_lightning.callbacks.RichModelSummary
        "learning_rate_logger": {
            "target": "pytorch_lightning.callbacks.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
                "log_momentum": True
            }
        },
        "image_logger": {
            "target": "utils.logger.CaptionImageLogger",
            "params": {
                "type": opt.logtype,
                "batch_frequency": 50, # 00,
                "max_images": 16,
                "clamp": True
            }
        },
    }
    callbacks_cfg = OmegaConf.create()
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg, modelckpt_cfg)
    trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
    if opt.activate_ddp_share:
        trainer_kwargs["strategy"] = "ddp_sharded"
    else:
        trainer_kwargs["strategy"] = DDPPlugin(find_unused_parameters=True)
    # trainer_kwargs["deterministic"] = True  # for reproducible

    trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)

    data = instantiate_from_config(config.data)
    data.prepare_data()

    if opt.gpus == -1:
        ngpu = torch.cuda.device_count()
    else:
        ngpu = len(opt.gpus.split(","))

    model.training_steps = len(data._train_dataloader()) * opt.max_epochs // ngpu
    model.steps_per_epoch = len(data._train_dataloader()) // ngpu
    model.max_epoch = opt.max_epochs

    # configure learning rate
    if "base_learning_rate" in config.model:
        print("Using base_learning_rate & Configure learning rate According to batch_size!")
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        # accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches or 1
        accumulate_grad_batches = 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        print("Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
            model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
    elif "learning_rate" in config.model:
        print("Using default learning_rate")
        model.learning_rate = config.model.learning_rate
    else:
        raise NotImplementedError("Please set learning rate!")
    
    if "min_learning_rate" in config.model:
        model.min_learning_rate = config.model.min_learning_rate
    else:
        model.min_learning_rate = 0.
    
    trainer.fit(model, data)
    trainer.save_checkpoint("{}/last.ckpt".format(ckptdir))