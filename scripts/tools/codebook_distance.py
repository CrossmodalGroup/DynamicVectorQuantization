import warnings
warnings.filterwarnings("ignore")

import os
import sys

sys.path.append(os.getcwd())
import argparse
import numpy as np

import torch
import torch.nn as nn
import torchvision
from data.imagenet_lmdb import Imagenet_LMDB
from data.ffhq_lmdb import FFHQ_LMDB
from omegaconf import OmegaConf
from utils.utils import instantiate_from_config
from modules.masked_quantization.tools import build_score_image
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--yaml_path", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--save_name", type=str, default="")

    return parser

if __name__ == "__main__":
    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    # init and save configs
    configs = OmegaConf.load(opt.yaml_path)
    # model
    model = instantiate_from_config(configs.model)
    state_dict = torch.load(opt.model_path)['state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.eval() # .cuda()

    try:
        embedding_data = model.quantize.embedding.weight.data
    except:
        try:
            embedding_data = model.quantize.codebook.weight.data
        except:
            embedding_data = model.quantizer.codebooks[0].weight.data
    
    min_distance_list = []
    for i in trange(embedding_data.size(0)):
        embedding_data_i = embedding_data[i].unsqueeze(0)
        l2_distance = nn.MSELoss(size_average=None, reduce=None, reduction='none')(embedding_data_i, embedding_data).mean(-1)
        l2_distance_sort, l1_distance_sort_order = l2_distance.sort(descending=False, dim=-1)
        
        min_distance_list.append(l2_distance_sort[1].item())
    
    min_distance_np = np.array(min_distance_list)
    print("maximum distance: ", min_distance_np.max())
    print("minumum distance: ", min_distance_np.min())
    print("average distance: ", min_distance_np.mean())
    
    print("len(min_distance_np): ", len(min_distance_list))
    
    # plt.hist(x=min_distance_np, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.hist(x=min_distance_np, bins=10, color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('minimum distance')
    plt.ylabel('number')
    # plt.title('My Very Own Histogram')
    plt.savefig("{}.png".format(opt.save_name))