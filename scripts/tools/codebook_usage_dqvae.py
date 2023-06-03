# The codebook usage is calculated as the percentage of
# used codes given a batch of 256 test images averaged over the entire test set.

import os
import sys

sys.path.append(os.getcwd())
import argparse

import torch
import torchvision
from data.imagenet_lmdb import Imagenet_LMDB
from data.ffhq_lmdb import FFHQ_LMDB
from omegaconf import OmegaConf
from utils.utils import instantiate_from_config
from modules.tokenizers.tools import build_score_image
from tqdm import tqdm

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--yaml_path", type=str, 
        default="logs/03-26T21-19-43_vqmask-50_coco_f16_1024/configs/03-26T21-19-43-project.yaml")
    parser.add_argument("--model_path", type=str, 
        default="logs/03-26T21-19-43_vqmask-50_coco_f16_1024/checkpoints/last.ckpt")

    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--dataset_type", type=str, default="ffhq")
    parser.add_argument("--codebook_size", type=int, default=1024)

    return parser


if __name__ == "__main__":
    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    # init and save configs
    configs = OmegaConf.load(opt.yaml_path)
    # model
    model = instantiate_from_config(configs.model)
    state_dict = torch.load(opt.model_path)['state_dict']
    model.load_state_dict(state_dict)
    model.eval().cuda()

    if opt.dataset_type == "ffhq":
        dset = FFHQ_LMDB(split="val", resolution=256, is_eval=True)
        dloader = torch.utils.data.DataLoader(dset, batch_size=opt.batch_size, num_workers=0, shuffle=False)
    elif opt.dataset_type == "imagenet":
        dset = Imagenet_LMDB(split="val", resolution=256, is_eval=True)
        dloader = torch.utils.data.DataLoader(dset, batch_size=opt.batch_size, num_workers=0, shuffle=False)
    
    with torch.no_grad():
        for i,data in tqdm(enumerate(dloader)):
            image = data["image"].float().cuda()
            # image = image.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)

            quant, emb_loss, info, _, _ = model.encode(image)

            min_encoding_indices = info[-1].view(-1)
            # print(min_encoding_indices.size())
            min_encoding_indices = min_encoding_indices.cpu().numpy().tolist()

            if i == 0:
                codebook_register = list(set(min_encoding_indices))
            else:
                codebook_register = list(set(codebook_register + min_encoding_indices))
    
    print(len(codebook_register))
    print("usage: ", 1 - len(codebook_register) / opt.codebook_size)