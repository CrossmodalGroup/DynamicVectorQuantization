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
from modules.multigrained.utils import draw_dual_grain_256res, draw_dual_grain_256res_color

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--dataset", type=str, default="imagenet")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num", type=int, default=10)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    hw = 32
    num = 256 // hw
    score_ratio = 0.7
    
    if opt.dataset == "imagenet-10":
        dataset = Imagenet_LMDB(split="val", resolution=256, is_eval=True)

        vqgan_config_path_f32 = "results/vqgan/vqgan_s1id05_imagenet_f32/configs/04-15T09-03-58-project.yaml"
        vqgan_model_path_f32 = "results/vqgan/vqgan_s1id05_imagenet_f32/checkpoints/last.ckpt"

        vqgan_config_path_f16 = "results/vqgan/vqgan_s1id03_imagenet_f16/configs/04-14T21-51-17-project.yaml"
        vqgan_model_path_f16 = "results/vqgan/vqgan_s1id03_imagenet_f16/checkpoints/last.ckpt"

        vqgan_config_path_f8 = "results/vqgan/vqgan_s1id04_imagenet_f8/configs/04-15T07-50-15-project.yaml"
        vqgan_model_path_f8 = "results/vqgan/vqgan_s1id04_imagenet_f8/epoch=9-val_rec_loss=0.2424.ckpt"

    elif opt.dataset == "ffhq":
        dataset = FFHQ_LMDB(split="train", resolution=256, is_eval=True)

        vqgan_config_path_f32 = "results/vqgan/vqgan_s1id02_ffhq_f32/configs/04-14T08-58-18-project.yaml"
        vqgan_model_path_f32 = "results/vqgan/vqgan_s1id02_ffhq_f32/checkpoints/last.ckpt"

        vqgan_config_path_f16 = "results/vqgan/vqgan_s1id00_ffhq_f16/configs/04-14T07-44-09-project.yaml"
        vqgan_model_path_f16 = "results/vqgan/vqgan_s1id00_ffhq_f16/checkpoints/last.ckpt"

        vqgan_config_path_f8 = "results/vqgan/vqgan_s1id01_ffhq_f8/configs/04-14T07-50-45-project.yaml"
        vqgan_model_path_f8 = "results/vqgan/vqgan_s1id01_ffhq_f8/checkpoints/last.ckpt"

        vqgan_triple_config_path_f_8_16_32 = "logs/08-21T14-48-05_dev6_tripleencode-Large_ffhq_f8_b-0125-0375-norm_kmeans_32d/08-21T14-48-05-project.yml"
        vqgan_triple_model_path_f8_16_32 = "logs/08-21T14-48-05_dev6_tripleencode-Large_ffhq_f8_b-0125-0375-norm_kmeans_32d/last.ckpt"

        vqgan_double_config_path_f_8_16 = "logs/08-19T05-22-51_dev6_dualencode_ffhq_f8_b05_norm_kmeans_32d/configs/08-19T05-22-51-project.yaml"
        vqgan_double_model_path_f8_16 = "logs/08-19T05-22-51_dev6_dualencode_ffhq_f8_b05_norm_kmeans_32d/checkpoints/last.ckpt"

    else:
        raise NotImplementedError()
    print("dataset length: ", len(dataset))
    dataloader = DataLoader(dataset, batch_size=1, num_workers=8, shuffle=True)

    vqgan_config_triple = OmegaConf.load(vqgan_triple_config_path_f_8_16_32)
    vqgan_model_triple = instantiate_from_config(vqgan_config_triple.model)
    vqgan_model_triple.load_state_dict(torch.load(vqgan_triple_model_path_f8_16_32)["state_dict"])
    vqgan_model_triple = vqgan_model_triple.cuda()

    vqgan_config_double_f_8_16 = OmegaConf.load(vqgan_double_config_path_f_8_16)
    vqgan_model_double_f_8_16 = instantiate_from_config(vqgan_config_double_f_8_16.model)
    vqgan_model_double_f_8_16.load_state_dict(torch.load(vqgan_double_model_path_f8_16)["state_dict"])
    vqgan_model_double_f_8_16 = vqgan_model_double_f_8_16.cuda()

    vqgan_config_f32 = OmegaConf.load(vqgan_config_path_f32)
    vqgan_model_f32 = instantiate_from_config(vqgan_config_f32.model)
    vqgan_model_f32.load_state_dict(torch.load(vqgan_model_path_f32)["state_dict"])
    vqgan_model_f32 = vqgan_model_f32.cuda()

    vqgan_config_f16 = OmegaConf.load(vqgan_config_path_f16)
    vqgan_model_f16 = instantiate_from_config(vqgan_config_f16.model)
    vqgan_model_f16.load_state_dict(torch.load(vqgan_model_path_f16)["state_dict"])
    vqgan_model_f16 = vqgan_model_f16.cuda()

    vqgan_config_f8 = OmegaConf.load(vqgan_config_path_f8)
    vqgan_model_f8 = instantiate_from_config(vqgan_config_f8.model)
    vqgan_model_f8.load_state_dict(torch.load(vqgan_model_path_f8)["state_dict"])
    vqgan_model_f8 = vqgan_model_f8.cuda()

    count = 0
    with torch.no_grad():
        for data_i, data in tqdm(enumerate(dataloader)):
            images = data["image"].cuda()

            rec_f32, _ = vqgan_model_f32(images)
            rec_f16, _ = vqgan_model_f16(images)
            rec_f8, _ = vqgan_model_f8(images)
            rec_triple, _, grain_indices, _ = vqgan_model_triple(images)
            rec_double_f_8_16, _, grain_indices, _ = vqgan_model_double_f_8_16(images)

            rec_triple_grain = draw_triple_grain_256res(images=images.clone(), indices=grain_indices)
            rec_triple_grain_color = draw_triple_grain_256res_color(images=images.clone(), indices=grain_indices, scaler=0.6)

            rec_double_f_8_16_grain = draw_dual_grain_256res(images=rec_triple.clone(), indices=grain_indices)
            rec_double_f_8_16_grain_color = draw_dual_grain_256res_color(images=rec_triple.clone(), indices=grain_indices, scaler=0.6)

            l1_loss_f32_list = []
            l1_loss_f16_list = []
            l1_loss_f8_list = []
            l1_loss_triple_list = []
            l1_loss_double_f_8_16_list = []
            for i in range(num):
                for j in range(num):
                    local_image = images[:, :, hw*i:hw*(i+1), hw*j:hw*(j+1)]

                    local_rec_f32 = rec_f32[:, :, hw*i:hw*(i+1), hw*j:hw*(j+1)]
                    l1_loss_f32 = nn.L1Loss()(local_rec_f32, local_image).item()

                    local_rec_f16 = rec_f16[:, :, hw*i:hw*(i+1), hw*j:hw*(j+1)]
                    l1_loss_f16 = nn.L1Loss()(local_rec_f16, local_image).item()

                    local_rec_f8 = rec_f8[:, :, hw*i:hw*(i+1), hw*j:hw*(j+1)]
                    l1_loss_f8 = nn.L1Loss()(local_rec_f8, local_image).item()

                    local_rec_triple = rec_triple[:, :, hw*i:hw*(i+1), hw*j:hw*(j+1)]
                    l1_loss_triple = nn.L1Loss()(local_rec_triple, local_image).item()

                    local_rec_double_f_8_16 = rec_double_f_8_16[:, :, hw*i:hw*(i+1), hw*j:hw*(j+1)]
                    l1_loss_double_f_8_16 = nn.L1Loss()(local_rec_double_f_8_16, local_image).item()

                    l1_loss_f32_list.append(l1_loss_f32)
                    l1_loss_f16_list.append(l1_loss_f16)
                    l1_loss_f8_list.append(l1_loss_f8)
                    l1_loss_triple_list.append(l1_loss_triple)
                    l1_loss_double_f_8_16_list.append(l1_loss_double_f_8_16)

            l1_loss_f32_tensor = torch.from_numpy(np.array(l1_loss_f32_list))
            l1_loss_f16_tensor = torch.from_numpy(np.array(l1_loss_f16_list))
            l1_loss_f8_tensor = torch.from_numpy(np.array(l1_loss_f8_list))
            l1_loss_triple_tensor = torch.from_numpy(np.array(l1_loss_triple_list))
            l1_loss_double_f_8_16_tensor = torch.from_numpy(np.array(l1_loss_double_f_8_16_list))


            l1_loss_f32_tensor = l1_loss_f32_tensor.view(8, 8).unsqueeze(0).unsqueeze(0).repeat_interleave(32, dim=-1).repeat_interleave(32, dim=-2)
            l1_loss_f16_tensor = l1_loss_f16_tensor.view(8, 8).unsqueeze(0).unsqueeze(0).repeat_interleave(32, dim=-1).repeat_interleave(32, dim=-2)
            l1_loss_f8_tensor = l1_loss_f8_tensor.view(8, 8).unsqueeze(0).unsqueeze(0).repeat_interleave(32, dim=-1).repeat_interleave(32, dim=-2)
            l1_loss_triple_tensor = l1_loss_triple_tensor.view(8, 8).unsqueeze(0).unsqueeze(0).repeat_interleave(32, dim=-1).repeat_interleave(32, dim=-2)
            l1_loss_double_f_8_16_tensor = l1_loss_double_f_8_16_tensor.view(8, 8).unsqueeze(0).unsqueeze(0).repeat_interleave(32, dim=-1).repeat_interleave(32, dim=-2)

            # normalization
            min_f32 = l1_loss_f32_tensor.min()
            max_f32 = l1_loss_f32_tensor.max()
            min_f16 = l1_loss_f16_tensor.min()
            max_f16 = l1_loss_f16_tensor.max()
            min_f8 = l1_loss_f8_tensor.min()
            max_f8 = l1_loss_f8_tensor.max()
            mix_triple = l1_loss_triple_tensor.min()
            max_triple = l1_loss_triple_tensor.max()
            min_double_f_8_16 = l1_loss_double_f_8_16_tensor.min()
            max_double_f_8_16 = l1_loss_double_f_8_16_tensor.max()

            min_value = min(min_f32, min_f16, min_f8, mix_triple, min_double_f_8_16)
            max_value = max(max_f32, max_f16, max_f8, max_triple, max_double_f_8_16)

            l1_loss_f32_tensor = (l1_loss_f32_tensor - min_value) / (max_value - min_value)
            l1_loss_f16_tensor = (l1_loss_f16_tensor - min_value) / (max_value - min_value)
            l1_loss_f8_tensor = (l1_loss_f8_tensor - min_value) / (max_value - min_value)
            l1_loss_triple_tensor = (l1_loss_triple_tensor - min_value) / (max_value - min_value)
            l1_loss_double_f_8_16_tensor = (l1_loss_double_f_8_16_tensor - min_value) / (max_value - min_value)

            image_errormap_f32 = build_score_image(images, l1_loss_f32_tensor, low_color="blue", high_color="red", scaler=score_ratio)
            torchvision.utils.save_image(image_errormap_f32, "temp/validate_idea_examples/{}_image_f32_errormap.png".format(data_i), normalize=False)
            rec_f32 = (rec_f32 * 0.5 + 0.5).clamp(0, 1)
            torchvision.utils.save_image(rec_f32, "temp/validate_idea_examples/{}_image_f32_rec.png".format(data_i), normalize=True)

            image_errormap_f16 = build_score_image(images, l1_loss_f16_tensor, low_color="blue", high_color="red", scaler=score_ratio)
            torchvision.utils.save_image(image_errormap_f16, "temp/validate_idea_examples/{}_image_f16_errormap.png".format(data_i), normalize=False)
            rec_f16 = (rec_f16 * 0.5 + 0.5).clamp(0, 1)
            torchvision.utils.save_image(rec_f16, "temp/validate_idea_examples/{}_image_f16_rec.png".format(data_i), normalize=True)

            image_errormap_f8 = build_score_image(images, l1_loss_f8_tensor, low_color="blue", high_color="red", scaler=score_ratio)
            torchvision.utils.save_image(image_errormap_f8, "temp/validate_idea_examples/{}_image_f8_errormap.png".format(data_i), normalize=False)
            rec_f8 = (rec_f8 * 0.5 + 0.5).clamp(0, 1)
            torchvision.utils.save_image(rec_f8, "temp/validate_idea_examples/{}_image_f8_rec.png".format(data_i), normalize=True)

            image_errormap_triple = build_score_image(images, l1_loss_triple_tensor, low_color="blue", high_color="red", scaler=score_ratio)
            torchvision.utils.save_image(image_errormap_triple, "temp/validate_idea_examples/{}_image_triple_errormap.png".format(data_i), normalize=False)
            rec_triple = (rec_triple * 0.5 + 0.5).clamp(0, 1)
            torchvision.utils.save_image(rec_triple, "temp/validate_idea_examples/{}_image_triple_rec.png".format(data_i), normalize=True)
            torchvision.utils.save_image(rec_triple_grain, "temp/validate_idea_examples/{}_image_triple_grain.png".format(data_i), normalize=True)
            torchvision.utils.save_image(rec_triple_grain_color, "temp/validate_idea_examples/{}_image_triple_grain_color.png".format(data_i), normalize=True)

            image_errormap_double_f_8_16 = build_score_image(images, l1_loss_double_f_8_16_tensor, low_color="blue", high_color="red", scaler=score_ratio)
            torchvision.utils.save_image(image_errormap_double_f_8_16, "temp/validate_idea_examples/{}_image_double_f_8_16_errormap.png".format(data_i), normalize=False)
            rec_double_f_8_16 = (rec_double_f_8_16 * 0.5 + 0.5).clamp(0, 1)
            torchvision.utils.save_image(rec_double_f_8_16, "temp/validate_idea_examples/{}_image_double_f_8_16_rec.png".format(data_i), normalize=True)
            torchvision.utils.save_image(rec_double_f_8_16_grain, "temp/validate_idea_examples/{}_image_double_f_8_16_grain.png".format(data_i), normalize=True)
            torchvision.utils.save_image(rec_double_f_8_16_grain_color, "temp/validate_idea_examples/{}_image_double_f_8_16_grain_color.png".format(data_i), normalize=True)


            torchvision.utils.save_image(images, "temp/validate_idea_examples/{}_image.png".format(data_i), normalize=True)

            count += 1
            if count >= opt.num:
                exit()