import argparse
import os
import sys

import torch
import torchvision
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np

sys.path.append(os.getcwd())

from data.imagenet import ImageNetValidation
from utils.utils import instantiate_from_config
from modules.dynamic_modules.utils import draw_dual_grain_256res_color

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--yaml_path", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_save_path", type=str, default="")

    return parser

if __name__ == "__main__":
    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    dataset = ImageNetValidation(config={"size" : 256, "is_eval" :True})
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
        dec, diff, grain_indices, gate, _ = model(images)

        sequence_length = 1 * (grain_indices == 0) + 4 * (grain_indices == 1)
        sequence_length = sequence_length.sum(-1).sum(-1)

        result_list += sequence_length.cpu().numpy().tolist()
        
        grain_map = draw_dual_grain_256res_color(images=images.clone(), indices=grain_indices, scaler=0.7)
        torchvision.utils.save_image(grain_map, "{}/grain_images_{}.png".format(opt.image_save_path, i))

    print("mean: ", np.mean(result_list))
    print("variance: ", np.var(result_list))
    print("max: ", max(result_list))
    print("min: ", min(result_list))