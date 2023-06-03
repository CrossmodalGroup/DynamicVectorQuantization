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
from modules.masked_quantization.tools import build_score_image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--yaml_path", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--title", type=str, default="PCA of codebook")
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
    model.load_state_dict(state_dict)
    model.eval().cuda()

    try:
        embedding_data = model.quantize.embedding.weight.data
    except:
        try:
            embedding_data = model.quantize.codebook.weight.data
        except:
            embedding_data = model.quantizer.codebooks[0].weight.data
    embedding_data = embedding_data.cpu().numpy()
    print(embedding_data)

    pca = PCA(n_components=2)  # 实例化
    pca = pca.fit(embedding_data)  # 拟合模型
    x_dr = pca.transform(embedding_data)  # 获取新矩阵

    print(x_dr.shape)

    plt.figure()  # 创建一个画布
    plt.scatter(x_dr[:,0],x_dr[:,1],c="red")  # plt.scatter(x_dr[y==0,0],x_dr[y==0,1],c="red",label = iris.target_names[0])
    # plt.legend()  # 显示图例
    plt.title("{}".format(opt.title))  # 显示标题
    plt.savefig("{}.png".format(opt.save_name))