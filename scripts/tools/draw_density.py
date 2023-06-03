import warnings
warnings.filterwarnings("ignore")

import os
import sys
from cv2 import normalize
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.getcwd())
import argparse

import torch
import torchvision
from data.ffhq_lmdb import FFHQ_LMDB
from data.imagenet_lmdb import Imagenet_LMDB
from omegaconf import OmegaConf
from utils.utils import instantiate_from_config
from models.stage1_masked.mqvae_adaptive_tensor import Entropy


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dataset_type", type=str, default="ffhq_val")

    return parser

if __name__ == "__main__":
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    
    entropy_model = Entropy(patch_size=256, image_width=256, image_height=256)

    if opt.dataset_type == "ffhq_val":
        dset = FFHQ_LMDB(split="val", resolution=256, is_eval=True)
    elif opt.dataset_type == "ffhq_train":
        dset = FFHQ_LMDB(split="train", resolution=256, is_eval=True)
    elif opt.dataset_type == "imagenet_val":
        dset = Imagenet_LMDB(split="val", resolution=256, is_eval=True)
    elif opt.dataset_type == "imagenet_train":
        dset = Imagenet_LMDB(split="train", resolution=256, is_eval=True)
    else:
        raise NotImplementedError()
    dloader = torch.utils.data.DataLoader(dset, batch_size=opt.batch_size, num_workers=0, shuffle=True)

    entropy_list = []
    with torch.no_grad():
        for i,data in tqdm(enumerate(dloader)):
            image = data["image"].float().cuda()
            image_entropy = entropy_model(image).view(-1)
            
            image_entropy_np = image_entropy.cpu().numpy()
            entropy_list += image_entropy_np.tolist()
            
            # if i == 2:
            #     break
    
    print(len(entropy_list))
    
    sns.distplot(
        image_entropy_np, bins=10, hist=True, kde=True, 
        hist_kws={'color':'g','histtype':'bar','alpha':0.4},
        kde_kws={'color':'r','linestyle':'-','linewidth':3,'alpha':0.7}
    )
    plt.savefig("entropy_kde_{}.png".format(opt.dataset_type))
    
    plt.figure()
    
    sns.distplot(
        image_entropy_np, bins=10, hist=True, kde=False, 
        hist_kws={'color':'g','histtype':'bar','alpha':0.4},
        kde_kws={'color':'r','linestyle':'-','linewidth':3,'alpha':0.7}
    )
    plt.savefig("entropy_hist_{}.png".format(opt.dataset_type))