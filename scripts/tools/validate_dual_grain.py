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
from modules.multigrained.utils import draw_dual_grain_256res, draw_dual_grain_256res_color

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--dataset", type=str, default="ffhq")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num", type=int, default=1)
    return parser

if __name__ == "__main__":
    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    hw = 16 # 32
    num = 256 // hw
    score_ratio = 0.6

    if opt.dataset == "imagenet":
        dataset = Imagenet_LMDB(split=opt.split, resolution=256, is_eval=True)

        vqgan_config_path_f16 = "results/vqgan/vqgan_s1id03_imagenet_f16/configs/04-14T21-51-17-project.yaml"
        vqgan_model_path_f16 = "results/vqgan/vqgan_s1id03_imagenet_f16/checkpoints/last.ckpt"

        vqgan_config_path_f8 = "results/vqgan/vqgan_s1id04_imagenet_f8/configs/04-15T07-50-15-project.yaml"
        vqgan_model_path_f8 = "results/vqgan/vqgan_s1id04_imagenet_f8/epoch=9-val_rec_loss=0.2424.ckpt"

        print("loading dynamic of p10")
        dynamic_p10_config = OmegaConf.load("logs_for_dynamic_quantization/10-21T07-34-47_S1_dual_D-simple_imgnet_b01-10_vq-old/configs/10-21T07-34-47-project.yaml")
        dynamic_p10_model = instantiate_from_config(dynamic_p10_config.model)
        dynamic_p10_model.load_state_dict(torch.load("logs_for_dynamic_quantization/10-21T07-34-47_S1_dual_D-simple_imgnet_b01-10_vq-old/checkpoints/epoch=15-val_rec_loss=0.4118.ckpt")["state_dict"])
        dynamic_p10_model = dynamic_p10_model.cuda()

        print("loading dynamic of p30")
        dynamic_p30_config = OmegaConf.load("logs_for_dynamic_quantization/10-09T11-45-38_S1_dual_D-simple_imgnet_b03-10_vq-old/configs/10-09T11-45-38-project.yaml")
        dynamic_p30_model = instantiate_from_config(dynamic_p30_config.model)
        dynamic_p30_model.load_state_dict(torch.load("logs_for_dynamic_quantization/10-09T11-45-38_S1_dual_D-simple_imgnet_b03-10_vq-old/epoch=12-val_rec_loss=0.3970.ckpt")["state_dict"])
        dynamic_p30_model = dynamic_p30_model.cuda()

        print("loading dynamic of p50")
        dynamic_p50_config = OmegaConf.load("logs_for_dynamic_quantization/10-09T13-31-50_S1_dual_D-simple_imgnet_b05-10_vq-old/configs/10-09T13-31-50-project.yaml")
        dynamic_p50_model = instantiate_from_config(dynamic_p50_config.model)
        dynamic_p50_model.load_state_dict(torch.load("logs_for_dynamic_quantization/10-09T13-31-50_S1_dual_D-simple_imgnet_b05-10_vq-old/epoch=15-val_rec_loss=0.3441.ckpt")["state_dict"])
        dynamic_p50_model = dynamic_p50_model.cuda()

        print("loading dynamic of p70")
        dynamic_p70_config = OmegaConf.load("logs_for_dynamic_quantization/10-10T19-39-49_S1_dual_D-simple_imgnet_b07-10_vq-old/configs/10-10T19-39-49-project.yaml")
        dynamic_p70_model = instantiate_from_config(dynamic_p70_config.model)
        dynamic_p70_model.load_state_dict(torch.load("logs_for_dynamic_quantization/10-10T19-39-49_S1_dual_D-simple_imgnet_b07-10_vq-old/epoch=9-val_rec_loss=0.3376.ckpt")["state_dict"])
        dynamic_p70_model = dynamic_p70_model.cuda()

    elif opt.dataset == "ffhq":
        dataset = FFHQ_LMDB(split=opt.split, resolution=256, is_eval=True)

        vqgan_config_path_f16 = "results/vqgan/vqgan_s1id00_ffhq_f16/configs/04-14T07-44-09-project.yaml"
        vqgan_model_path_f16 = "results/vqgan/vqgan_s1id00_ffhq_f16/checkpoints/last.ckpt"

        vqgan_config_path_f8 = "results/vqgan/vqgan_s1id01_ffhq_f8/configs/04-14T07-50-45-project.yaml"
        vqgan_model_path_f8 = "results/vqgan/vqgan_s1id01_ffhq_f8/checkpoints/last.ckpt"

        print("loading dynamic of p10")
        dynamic_p10_config = OmegaConf.load("logs/09-16T09-20-56_S1_dual_D-simple_ffhq_b01-10/configs/09-16T09-20-56-project.yaml")
        dynamic_p10_model = instantiate_from_config(dynamic_p10_config.model)
        dynamic_p10_model.load_state_dict(torch.load("logs/09-16T09-20-56_S1_dual_D-simple_ffhq_b01-10/checkpoints/last.ckpt")["state_dict"])
        dynamic_p10_model = dynamic_p10_model.cuda()

        print("loading dynamic of p30")
        dynamic_p30_config = OmegaConf.load("logs/09-15T22-40-32_S1_dual_D-simple_ffhq_b03-10/configs/09-15T22-40-32-project.yaml")
        dynamic_p30_model = instantiate_from_config(dynamic_p30_config.model)
        dynamic_p30_model.load_state_dict(torch.load("logs/09-15T22-40-32_S1_dual_D-simple_ffhq_b03-10/checkpoints/epoch=138-val_rec_loss=0.2481.ckpt")["state_dict"])
        dynamic_p30_model = dynamic_p30_model.cuda()

        print("loading dynamic of p50")
        dynamic_p50_config = OmegaConf.load("logs/09-15T23-33-43_S1_dual_D-simple_ffhq_b05-10/configs/09-15T23-33-43-project.yaml")
        dynamic_p50_model = instantiate_from_config(dynamic_p50_config.model)
        dynamic_p50_model.load_state_dict(torch.load("logs/09-15T23-33-43_S1_dual_D-simple_ffhq_b05-10/checkpoints/epoch=147-val_rec_loss=0.2239.ckpt")["state_dict"])
        dynamic_p50_model = dynamic_p50_model.cuda()

        print("loading dynamic of p70")
        dynamic_p70_config = OmegaConf.load("logs/09-15T20-26-46_S1_dual_D-simple_ffhq_b07-10/configs/09-15T20-26-46-project.yaml")
        dynamic_p70_model = instantiate_from_config(dynamic_p70_config.model)
        dynamic_p70_model.load_state_dict(torch.load("logs/09-15T20-26-46_S1_dual_D-simple_ffhq_b07-10/checkpoints/epoch=148-val_rec_loss=0.1978.ckpt")["state_dict"])
        dynamic_p70_model = dynamic_p70_model.cuda()

    else:
        raise NotImplementedError()
    print("dataset length: ", len(dataset))
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)

    print("loading vqgan f16")
    vqgan_config_f16 = OmegaConf.load(vqgan_config_path_f16)
    vqgan_model_f16 = instantiate_from_config(vqgan_config_f16.model)
    vqgan_model_f16.load_state_dict(torch.load(vqgan_model_path_f16)["state_dict"])
    vqgan_model_f16 = vqgan_model_f16.cuda()

    print("loading vqgan f8")
    vqgan_config_f8 = OmegaConf.load(vqgan_config_path_f8)
    vqgan_model_f8 = instantiate_from_config(vqgan_config_f8.model)
    vqgan_model_f8.load_state_dict(torch.load(vqgan_model_path_f8)["state_dict"])
    vqgan_model_f8 = vqgan_model_f8.cuda()

    count = 0
    with torch.no_grad():
        for data_i, data in tqdm(enumerate(dataloader)):
            images = data["image"].cuda()

            rec_f16, _ = vqgan_model_f16(images)
            rec_f8, _ = vqgan_model_f8(images)

            rec_p10, _, grain_indices_p10, _ = dynamic_p10_model(images)
            rec_p10_grain = draw_dual_grain_256res(images=images.clone(), indices=grain_indices_p10)
            rec_p10_grain_color = draw_dual_grain_256res_color(images=images.clone(), indices=grain_indices_p10, scaler=score_ratio)

            rec_p30, _, grain_indices_p30, _ = dynamic_p30_model(images)
            rec_p30_grain = draw_dual_grain_256res(images=images.clone(), indices=grain_indices_p30)
            rec_p30_grain_color = draw_dual_grain_256res_color(images=images.clone(), indices=grain_indices_p30, scaler=score_ratio)

            rec_p50, _, grain_indices_p50, _ = dynamic_p50_model(images)
            rec_p50_grain = draw_dual_grain_256res(images=images.clone(), indices=grain_indices_p50)
            rec_p50_grain_color = draw_dual_grain_256res_color(images=images.clone(), indices=grain_indices_p50, scaler=score_ratio)

            rec_p70, _, grain_indices_p70, _ = dynamic_p70_model(images)
            rec_p70_grain = draw_dual_grain_256res(images=images.clone(), indices=grain_indices_p70)
            rec_p70_grain_color = draw_dual_grain_256res_color(images=images.clone(), indices=grain_indices_p70, scaler=score_ratio)


            l1_loss_f16_list = []
            l1_loss_f8_list = []
            l1_loss_p10_list = []
            l1_loss_p30_list = []
            l1_loss_p50_list = []
            l1_loss_p70_list = []
            for i in range(num):
                for j in range(num):
                    local_image = images[:, :, hw*i:hw*(i+1), hw*j:hw*(j+1)]

                    local_rec_f16 = rec_f16[:, :, hw*i:hw*(i+1), hw*j:hw*(j+1)]
                    l1_loss_f16 = nn.L1Loss()(local_rec_f16, local_image).item()

                    local_rec_f8 = rec_f8[:, :, hw*i:hw*(i+1), hw*j:hw*(j+1)]
                    l1_loss_f8 = nn.L1Loss()(local_rec_f8, local_image).item()

                    local_rec_p10 = rec_p10[:, :, hw*i:hw*(i+1), hw*j:hw*(j+1)]
                    l1_loss_p10 = nn.L1Loss()(local_rec_p10, local_image).item()

                    local_rec_p30 = rec_p30[:, :, hw*i:hw*(i+1), hw*j:hw*(j+1)]
                    l1_loss_p30 = nn.L1Loss()(local_rec_p30, local_image).item()

                    local_rec_p50 = rec_p50[:, :, hw*i:hw*(i+1), hw*j:hw*(j+1)]
                    l1_loss_p50 = nn.L1Loss()(local_rec_p50, local_image).item()

                    local_rec_p70 = rec_p70[:, :, hw*i:hw*(i+1), hw*j:hw*(j+1)]
                    l1_loss_p70 = nn.L1Loss()(local_rec_p70, local_image).item()

                    l1_loss_f16_list.append(l1_loss_f16)
                    l1_loss_f8_list.append(l1_loss_f8)
                    l1_loss_p10_list.append(l1_loss_p10)
                    l1_loss_p30_list.append(l1_loss_p30)
                    l1_loss_p50_list.append(l1_loss_p50)
                    l1_loss_p70_list.append(l1_loss_p70)

            l1_loss_f16_tensor = torch.from_numpy(np.array(l1_loss_f16_list))
            l1_loss_f8_tensor = torch.from_numpy(np.array(l1_loss_f8_list))
            l1_loss_p10_tensor = torch.from_numpy(np.array(l1_loss_p10_list))
            l1_loss_p30_tensor = torch.from_numpy(np.array(l1_loss_p30_list))
            l1_loss_p50_tensor = torch.from_numpy(np.array(l1_loss_p50_list))
            l1_loss_p70_tensor = torch.from_numpy(np.array(l1_loss_p70_list))

            l1_loss_f16_tensor = l1_loss_f16_tensor.view(num, num).unsqueeze(0).unsqueeze(0).repeat_interleave(hw, dim=-1).repeat_interleave(hw, dim=-2)
            l1_loss_f8_tensor = l1_loss_f8_tensor.view(num, num).unsqueeze(0).unsqueeze(0).repeat_interleave(hw, dim=-1).repeat_interleave(hw, dim=-2)
            l1_loss_p10_tensor = l1_loss_p10_tensor.view(num, num).unsqueeze(0).unsqueeze(0).repeat_interleave(hw, dim=-1).repeat_interleave(hw, dim=-2)
            l1_loss_p30_tensor = l1_loss_p30_tensor.view(num, num).unsqueeze(0).unsqueeze(0).repeat_interleave(hw, dim=-1).repeat_interleave(hw, dim=-2)
            l1_loss_p50_tensor = l1_loss_p50_tensor.view(num, num).unsqueeze(0).unsqueeze(0).repeat_interleave(hw, dim=-1).repeat_interleave(hw, dim=-2)
            l1_loss_p70_tensor = l1_loss_p70_tensor.view(num, num).unsqueeze(0).unsqueeze(0).repeat_interleave(hw, dim=-1).repeat_interleave(hw, dim=-2)


            # normalization
            min_f16 = l1_loss_f16_tensor.min()
            max_f16 = l1_loss_f16_tensor.max()
            min_f8 = l1_loss_f8_tensor.min()
            max_f8 = l1_loss_f8_tensor.max()

            min_value = min(min_f16, min_f8, l1_loss_p10_tensor.min(), l1_loss_p30_tensor.min(), l1_loss_p50_tensor.min(), l1_loss_p70_tensor.min())
            max_value = max(max_f16, max_f8, l1_loss_p10_tensor.max(), l1_loss_p30_tensor.max(), l1_loss_p50_tensor.max(), l1_loss_p70_tensor.max())

            l1_loss_f16_tensor = (l1_loss_f16_tensor - min_value) / (max_value - min_value)
            l1_loss_f8_tensor = (l1_loss_f8_tensor - min_value) / (max_value - min_value)
            l1_loss_p10_tensor = (l1_loss_p10_tensor - min_value) / (max_value - min_value)
            l1_loss_p30_tensor = (l1_loss_p30_tensor - min_value) / (max_value - min_value)
            l1_loss_p50_tensor = (l1_loss_p50_tensor - min_value) / (max_value - min_value)
            l1_loss_p70_tensor = (l1_loss_p70_tensor - min_value) / (max_value - min_value)

            image_errormap_f16 = build_score_image(images, l1_loss_f16_tensor, low_color="blue", high_color="red", scaler=score_ratio)
            torchvision.utils.save_image(image_errormap_f16, "temp/validate_dynamic_dual_works/{}_image_f16_errormap.png".format(data_i), normalize=False)
            rec_f16 = (rec_f16 * 0.5 + 0.5).clamp(0, 1)
            torchvision.utils.save_image(rec_f16, "temp/validate_dynamic_dual_works/{}_image_f16_rec.png".format(data_i), normalize=True)

            image_errormap_f8 = build_score_image(images, l1_loss_f8_tensor, low_color="blue", high_color="red", scaler=score_ratio)
            torchvision.utils.save_image(image_errormap_f8, "temp/validate_dynamic_dual_works/{}_image_f8_errormap.png".format(data_i), normalize=False)
            rec_f8 = (rec_f8 * 0.5 + 0.5).clamp(0, 1)
            torchvision.utils.save_image(rec_f8, "temp/validate_dynamic_dual_works/{}_image_f8_rec.png".format(data_i), normalize=True)

            image_errormap_p10 = build_score_image(images, l1_loss_p10_tensor, low_color="blue", high_color="red", scaler=score_ratio)
            torchvision.utils.save_image(image_errormap_p10, "temp/validate_dynamic_dual_works/{}_image_p10_errormap.png".format(data_i), normalize=False)
            rec_p10 = (rec_p10 * 0.5 + 0.5).clamp(0, 1)
            torchvision.utils.save_image(rec_p10, "temp/validate_dynamic_dual_works/{}_image_p10_rec.png".format(data_i), normalize=True)
            torchvision.utils.save_image(rec_p10_grain, "temp/validate_dynamic_dual_works/{}_image_p10_grain.png".format(data_i), normalize=True)
            torchvision.utils.save_image(rec_p10_grain_color, "temp/validate_dynamic_dual_works/{}_image_p10_grain_color.png".format(data_i), normalize=True)

            image_errormap_p30 = build_score_image(images, l1_loss_p30_tensor, low_color="blue", high_color="red", scaler=score_ratio)
            torchvision.utils.save_image(image_errormap_p30, "temp/validate_dynamic_dual_works/{}_image_p30_errormap.png".format(data_i), normalize=False)
            rec_p30 = (rec_p30 * 0.5 + 0.5).clamp(0, 1)
            torchvision.utils.save_image(rec_p30, "temp/validate_dynamic_dual_works/{}_image_p30_rec.png".format(data_i), normalize=True)
            torchvision.utils.save_image(rec_p30_grain, "temp/validate_dynamic_dual_works/{}_image_p30_grain.png".format(data_i), normalize=True)
            torchvision.utils.save_image(rec_p30_grain_color, "temp/validate_dynamic_dual_works/{}_image_p30_grain_color.png".format(data_i), normalize=True)

            image_errormap_p50 = build_score_image(images, l1_loss_p50_tensor, low_color="blue", high_color="red", scaler=score_ratio)
            torchvision.utils.save_image(image_errormap_p50, "temp/validate_dynamic_dual_works/{}_image_p50_errormap.png".format(data_i), normalize=False)
            rec_p50 = (rec_p50 * 0.5 + 0.5).clamp(0, 1)
            torchvision.utils.save_image(rec_p50, "temp/validate_dynamic_dual_works/{}_image_p50_rec.png".format(data_i), normalize=True)
            torchvision.utils.save_image(rec_p50_grain, "temp/validate_dynamic_dual_works/{}_image_p50_grain.png".format(data_i), normalize=True)
            torchvision.utils.save_image(rec_p50_grain_color, "temp/validate_dynamic_dual_works/{}_image_p50_grain_color.png".format(data_i), normalize=True)

            image_errormap_p70 = build_score_image(images, l1_loss_p70_tensor, low_color="blue", high_color="red", scaler=score_ratio)
            torchvision.utils.save_image(image_errormap_p50, "temp/validate_dynamic_dual_works/{}_image_p70_errormap.png".format(data_i), normalize=False)
            rec_p50 = (rec_p50 * 0.5 + 0.5).clamp(0, 1)
            torchvision.utils.save_image(rec_p70, "temp/validate_dynamic_dual_works/{}_image_p70_rec.png".format(data_i), normalize=True)
            torchvision.utils.save_image(rec_p70_grain, "temp/validate_dynamic_dual_works/{}_image_p70_grain.png".format(data_i), normalize=True)
            torchvision.utils.save_image(rec_p70_grain_color, "temp/validate_dynamic_dual_works/{}_image_p70_grain_color.png".format(data_i), normalize=True)

            torchvision.utils.save_image(images, "temp/validate_dynamic_dual_works/{}_image.png".format(data_i), normalize=True)

            count += 1
            if count >= opt.num:
                exit()