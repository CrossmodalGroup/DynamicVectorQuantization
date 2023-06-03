from locale import normalize
import os
import sys

sys.path.append(os.getcwd())
import argparse
import pickle

import torch
from omegaconf import OmegaConf
from tqdm import trange
from utils.utils import instantiate_from_config

import datetime
import pytz
import torchvision

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--yaml_path", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--sample_with_fixed_pos", action="store_true", default=False)

    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=300)
    parser.add_argument("--top_k_pos", type=int, default=1024) # 50
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_p_pos", type=float, default=1.0)
    parser.add_argument("--sample_num", type=int, default=5000)

    return parser

if __name__ == "__main__":
    Shanghai = pytz.timezone("Asia/Shanghai")
    now = datetime.datetime.now().astimezone(Shanghai).strftime("%m-%dT%H-%M-%S")

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    
    save_path = opt.model_path.replace(".ckpt", "") + "_{}_Num-{}/".format(now, opt.sample_num)
    if opt.sample_with_fixed_pos:
        save_path_image = save_path + "fixed_TopK-{}-{}_TopP-{}-{}_Temp-{}_image".format(opt.top_k, opt.top_k_pos, opt.top_p, opt.top_p_pos, opt.temperature)
        save_path_pickle = save_path + "fixed_TopK-{}-{}_TopP-{}-{}_Temp-{}_pickle".format(opt.top_k, opt.top_k_pos, opt.top_p, opt.top_p_pos, opt.temperature)
        os.makedirs(save_path_image, exist_ok=True)
    else:
        save_path_image = save_path + "TopK-{}-{}_TopP-{}-{}_Temp-{}_image".format(opt.top_k, opt.top_k_pos, opt.top_p, opt.top_p_pos, opt.temperature)
        save_path_pickle = save_path + "TopK-{}-{}_TopP-{}-{}_Temp-{}_pickle".format(opt.top_k, opt.top_k_pos, opt.top_p, opt.top_p_pos, opt.temperature)
        os.makedirs(save_path_image, exist_ok=True)

    # init and save configs
    configs = OmegaConf.load(opt.yaml_path)
    # model
    model = instantiate_from_config(configs.model)
    state_dict = torch.load(opt.model_path)['state_dict']
    model.load_state_dict(state_dict)
    model.eval().cuda()
    
    if opt.sample_num % opt.batch_size == 0:
        total_batch = opt.sample_num // opt.batch_size
    else:
        total_batch = opt.sample_num // opt.batch_size + 1
    
    batch_size = opt.batch_size
    for i in trange(total_batch):
        if opt.sample_num % opt.batch_size != 0 and i == total_batch - 1:
            batch_size = opt.sample_num % opt.batch_size
        x0 = torch.randn(batch_size, ).cuda()
        c_coarse, c_fine, c_pos_coarse, c_pos_fine, c_seg_coarse, c_seg_fine = model.encode_to_c(x0)

        if opt.sample_with_fixed_pos:
            coarse_content, fine_content, coarse_position, fine_position = model.sample_from_scratch(
                c_coarse, c_fine, c_pos_coarse, c_pos_fine, c_seg_coarse, c_seg_fine, 
                temperature = opt.temperature,
                sample = True,
                top_k = opt.top_k,
                top_p = opt.top_p,
                top_k_pos = opt.top_k_pos,
                top_p_pos = opt.top_p_pos,
                process = True,
                fix_fine_position = True,
            )
            samples = model.decode_to_img(coarse_content, fine_content, coarse_position, fine_position)
            sample = torch.clamp((samples * 0.5 + 0.5), 0, 1)

            for batch_i in range(batch_size):
                torchvision.utils.save_image(samples[batch_i], "{}/batch_{}_{}.png".format(save_path_image, i, batch_i), normalize=True)
        else:
            coarse_content, fine_content, coarse_position, fine_position = model.sample_from_scratch(
                    c_coarse, c_fine, c_pos_coarse, c_pos_fine, c_seg_coarse, c_seg_fine, 
                    temperature = opt.temperature,
                    sample = True,
                    top_k = opt.top_k,
                    top_p = opt.top_p,
                    top_k_pos = opt.top_k_pos,
                    top_p_pos = opt.top_p_pos,
                    process = True,
                    fix_fine_position = False,
                )
            samples = model.decode_to_img(coarse_content, fine_content, coarse_position, fine_position)
            samples = torch.clamp((samples * 0.5 + 0.5), 0, 1)
            for batch_i in range(batch_size):
                torchvision.utils.save_image(samples[batch_i], "{}/batch_{}_{}.png".format(save_path_image, i, batch_i), normalize=True)