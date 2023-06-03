import torch
import torch.nn as nn

import os, sys
sys.path.append(os.getcwd())

from utils.utils import instantiate_from_config
from torch.utils.data import DataLoader

from data.imagenet_lmdb import Imagenet_LMDB
from data.ffhq_lmdb import FFHQ_LMDB
from omegaconf import OmegaConf
import torchvision
from tqdm import tqdm
import argparse
import numpy as np

from modules.tokenizers.tools import build_score_image
from modules.multigrained.utils import draw_triple_grain_256res_color, draw_triple_grain_256res

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--dataset", type=str, default="ffhq")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num", type=int, default=1)
    return parser

if __name__ == "__main__":
    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    if opt.dataset == "imagenet":
        dataset = Imagenet_LMDB(split="val", resolution=256, is_eval=True)

    elif opt.dataset == "ffhq":
        dataset = FFHQ_LMDB(split="train", resolution=256, is_eval=True)

        vqgan_triple_config_path_f_8_16_32 = "logs/08-21T14-48-05_dev6_tripleencode-Large_ffhq_f8_b-0125-0375-norm_kmeans_32d/08-21T14-48-05-project.yml"
        vqgan_triple_model_path_f8_16_32 = "logs/08-21T14-48-05_dev6_tripleencode-Large_ffhq_f8_b-0125-0375-norm_kmeans_32d/last.ckpt"

    else:
        raise NotImplementedError()
    print("dataset length: ", len(dataset))
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=8, shuffle=True)

    vqgan_config_triple = OmegaConf.load(vqgan_triple_config_path_f_8_16_32)
    vqgan_model_triple = instantiate_from_config(vqgan_config_triple.model)
    vqgan_model_triple.load_state_dict(torch.load(vqgan_triple_model_path_f8_16_32)["state_dict"])
    vqgan_model_triple = vqgan_model_triple.cuda()

    with torch.no_grad():
        for data_i, data in tqdm(enumerate(dataloader)):
            images = data["image"].cuda()

            quant, emb_loss, info, grain_indices, gate = vqgan_model_triple.encode(images)
            indices = info[2]
            image_grain_color = draw_triple_grain_256res_color(images, indices=grain_indices, scaler=0.6)

            grain_indices = grain_indices.repeat_interleave(4, dim=-1).repeat_interleave(4, dim=-2).unsqueeze(-1)

            embeddings, _ = vqgan_model_triple.get_code_emb_with_depth(indices)
            empty_embeddings = torch.zeros_like(embeddings)

            embeddings_only_0 = torch.where(grain_indices==0, embeddings, empty_embeddings)
            embeddings_only_1 = torch.where(grain_indices==1, embeddings, empty_embeddings)
            embeddings_only_2 = torch.where(grain_indices==2, embeddings, empty_embeddings)

            rec = vqgan_model_triple.decode(embeddings.permute(0, 3, 1, 2))
            rec_0 = vqgan_model_triple.decode(embeddings_only_0.permute(0, 3, 1, 2))
            rec_1 = vqgan_model_triple.decode(embeddings_only_1.permute(0, 3, 1, 2))
            rec_2 = vqgan_model_triple.decode(embeddings_only_2.permute(0, 3, 1, 2))

            torchvision.utils.save_image(images, "real.png", normalize=True, nrow=2)
            torchvision.utils.save_image(rec, "rec.png", normalize=True, nrow=2)
            torchvision.utils.save_image(rec_0, "rec_0.png", normalize=True, nrow=2)
            torchvision.utils.save_image(rec_1, "rec_1.png", normalize=True, nrow=2)
            torchvision.utils.save_image(rec_2, "rec_2.png", normalize=True, nrow=2)
            torchvision.utils.save_image(image_grain_color, "image_grain_color.png", normalize=True, nrow=2)
            exit()