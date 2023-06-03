import argparse
import os
import sys

import torch
import torchvision
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np

sys.path.append(os.getcwd())

from data.ffhq_lmdb import FFHQ_LMDB
from utils.utils import instantiate_from_config


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--yaml_path", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--dataset", type=str, default="ffhq")
    parser.add_argument("--batch_size", type=int, default=4)

    return parser

if __name__ == "__main__":
    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    if opt.dataset == "ffhq":
        dataset = FFHQ_LMDB(split="val", resolution=256, is_eval=True)
    else:
        raise NotImplementedError()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, num_workers=1)

    # init and save configs
    configs = OmegaConf.load(opt.yaml_path)
    # model
    model = instantiate_from_config(configs.model)
    state_dict = torch.load(opt.model_path)['state_dict']
    model.load_state_dict(state_dict)
    model.eval().cuda()

    result_list = []
    for i, data in tqdm(enumerate(dataloader)):
        images = data["image"].float().cuda()
        dec, diff, grain_indices, gate = model(images)

        sequence_length = 1 * (grain_indices == 0) + 4 * (grain_indices == 1)
        sequence_length = sequence_length.sum(-1).sum(-1)

        result_list += sequence_length.cpu().numpy().tolist()

    print("mean: ", np.mean(result_list))
    print("variance: ", np.var(result_list))
    print("max: ", max(result_list))
    print("min: ", min(result_list))