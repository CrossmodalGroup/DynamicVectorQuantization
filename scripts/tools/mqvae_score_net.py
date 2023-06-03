import os, sys

sys.path.append(os.getcwd())

from omegaconf import OmegaConf
import torch
import torchvision 
from utils.utils import instantiate_from_config
import argparse
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from modules.masked_quantization.tools import build_score_image
import scipy.stats as ss
from tqdm import tqdm

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--yaml_path", type=str, default="logs/results/mqvae/mqvae_s1id09_75_imagenet_f8_T6bsa/configs/04-29T09-15-00-project.yaml")
    parser.add_argument("--model_path", type=str, default="logs/results/mqvae/mqvae_s1id09_75_imagenet_f8_T6bsa/checkpoints/epoch=7-val_rec_loss=0.4564.ckpt")
    # parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--vqvae_yaml", type=str, default="logs/results/vqgan/vqgan_s1id04_imagenet_f8/configs/04-15T07-50-15-project.yaml")
    parser.add_argument("--vqvae_model", type=str, default="logs/results/vqgan/vqgan_s1id04_imagenet_f8/epoch=9-val_rec_loss=0.2424.ckpt")
    return parser

if __name__ == "__main__":
    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    yaml_path = opt.yaml_path
    model_path = opt.model_path

    # init and save configs
    config = OmegaConf.load(yaml_path)
    
    # model
    model = instantiate_from_config(config.model)  # .cuda()
    model.load_state_dict(torch.load(model_path)["state_dict"], strict=False)
    model.cuda()
    
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    
    val_dloader = data.val_dataloader()
    train_dloader = data.train_dataloader()

    vqvae_config = OmegaConf.load(opt.vqvae_yaml)
    # model
    vqvae_model = instantiate_from_config(vqvae_config.model)  # .cuda()
    vqvae_model.load_state_dict(torch.load(opt.vqvae_model)["state_dict"], strict=False)
    vqvae_model.cuda()
    
    correlation_list = []
    p_list = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(train_dloader)):
            images = data["image"].cuda()
            # images = data["image"].cuda().float().permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
            dec, diff, preforward_dict = model(images)

            image_values_map = preforward_dict["image_features_avg_values"].unsqueeze(1)
            image_values_map = F.interpolate(image_values_map, scale_factor=8, mode="nearest")
            # print(image_values_map.size())
            # print(preforward_dict["score_map"].size())

            original_images = images * 0.5 + 0.5
            original_images = torch.clamp(original_images, 0, 1)
            # masked_images = preforward_dict["binary_map"] * 0.5 + 0.5
            # masked_images = torch.clamp(masked_images, 0, 1) * original_images
            masked_images = images * preforward_dict["binary_map"]

            score_map = build_score_image(images, preforward_dict["score_map"], low_color="blue", high_color="red", scaler=0.9)

            # # 计算两组数据的相关性
            # score_numpy = preforward_dict["predicted_score"][0].cpu().numpy()
            # image_value_numpy = rearrange(preforward_dict["image_features_avg_values"], "B H W -> B (H W)")[0].cpu().numpy()
            # # correlation = np.corrcoef(score_numpy, image_value_numpy)
            # correlation = ss.pearsonr(image_value_numpy, score_numpy)

            # # print(correlation)
            # correlation_list.append(correlation[0])
            # p_list.append(correlation[1])

            # if i > 1000:
            #     break

            # print(correlation_list, p_list)
            # exit()
            
            torchvision.utils.save_image(dec, "tests/rec.png", normalize=True, nrow=4)
            torchvision.utils.save_image(images, "tests/real.png", normalize=True, nrow=4)
            torchvision.utils.save_image(masked_images, "tests/binary_mask.png", normalize=True, nrow=4)
            torchvision.utils.save_image(score_map, "tests/score_map.png", normalize=True, nrow=4)
            torchvision.utils.save_image(image_values_map, "tests/image_values_map.png", normalize=True, nrow=4)
            
            _, _, _, vqvae_image_values = vqvae_model.encode2(images)
            vqvae_image_values = vqvae_image_values.mean(1).unsqueeze(1)
            vqvae_image_values = F.interpolate(vqvae_image_values, scale_factor=8, mode="nearest")
            torchvision.utils.save_image(vqvae_image_values, "tests/vqvae_image_values_map.png", normalize=True, nrow=4)

            exit()

    # mean_correlation = np.array(correlation_list).mean()
    # mean_p = np.array(p_list).mean()
    # print(mean_correlation, mean_p)