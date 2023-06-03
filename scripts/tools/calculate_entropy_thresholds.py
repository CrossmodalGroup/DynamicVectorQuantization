import os
import sys

sys.path.append(os.getcwd())
import argparse

import torch
from einops import rearrange
from torch import nn, Tensor
import torchvision
from data.imagenet_lmdb import Imagenet_LMDB
from data.ffhq_lmdb import FFHQ_LMDB
from tqdm import tqdm
import numpy as np
np.set_printoptions(suppress=True)
import json

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--dataset_type", type=str, default="ffhq")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=256)
    return parser

class Entropy(nn.Sequential):
    def __init__(self, patch_size, image_width, image_height):
        super(Entropy, self).__init__()
        self.width = image_width
        self.height = image_height
        self.psize = patch_size
        # number of patches per image
        self.patch_num = int(self.width * self.height / self.psize ** 2)
        self.hw = int(self.width // self.psize)
        # unfolding image to non overlapping patches
        self.unfold = torch.nn.Unfold(kernel_size=(self.psize, self.psize), stride=self.psize)

    def entropy(self, values: torch.Tensor, bins: torch.Tensor, sigma: torch.Tensor, batch: int) -> torch.Tensor:
        """Function that calculates the entropy using marginal probability distribution function of the input tensor
            based on the number of histogram bins.
        Args:
            values: shape [BxNx1].
            bins: shape [NUM_BINS].
            sigma: shape [1], gaussian smoothing factor.
            batch: int, size of the batch
        Returns:
            torch.Tensor:
        """
        epsilon = 1e-40
        values = values.unsqueeze(2)
        residuals = values - bins.unsqueeze(0).unsqueeze(0)
        kernel_values = torch.exp(-0.5 * (residuals / sigma).pow(2))

        pdf = torch.mean(kernel_values, dim=1)
        normalization = torch.sum(pdf, dim=1).unsqueeze(1) + epsilon
        pdf = pdf / normalization + epsilon
        entropy = - torch.sum(pdf * torch.log(pdf), dim=1)
        entropy = entropy.reshape((batch, -1))
        entropy = rearrange(entropy, "B (H W) -> B H W", H=self.hw, W=self.hw)
        return entropy

    def forward(self, inputs: Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        gray_images = 0.2989 * inputs[:, 0:1, :, :] + 0.5870 * inputs[:, 1:2, :, :] + 0.1140 * inputs[:, 2:, :, :]

        # create patches of size (batch x patch_size*patch_size x h*w/ (patch_size*patch_size))
        unfolded_images = self.unfold(gray_images)
        # reshape to (batch * h*w/ (patch_size*patch_size) x (patch_size*patch_size)
        unfolded_images = unfolded_images.transpose(1, 2)
        unfolded_images = torch.reshape(unfolded_images.unsqueeze(2),
                                        (unfolded_images.shape[0] * self.patch_num, unfolded_images.shape[2]))

        entropy = self.entropy(unfolded_images, bins=torch.linspace(0, 1, 32).to(device=inputs.device),
                               sigma=torch.tensor(0.01), batch=batch_size)

        return entropy

if __name__ == "__main__":
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    
    if opt.dataset_type == "ffhq":
        dset = FFHQ_LMDB(split=opt.split, resolution=opt.image_size, is_eval=True)
        dloader = torch.utils.data.DataLoader(dset, batch_size=opt.batch_size, num_workers=0, shuffle=False)
    elif opt.dataset_type == "imagenet":
        dset = Imagenet_LMDB(split=opt.split, resolution=opt.image_size, is_eval=True)
        dloader = torch.utils.data.DataLoader(dset, batch_size=opt.batch_size, num_workers=0, shuffle=False)
    
    model = Entropy(opt.patch_size, opt.image_size, opt.image_size).cuda()

    with torch.no_grad():
        for i, data in tqdm(enumerate(dloader)):
            image = data["image"].cuda()
            if i == 0:
                entropy_numpy = model(image).view(-1).cpu().numpy()
            else:
                entropy_numpy = np.concatenate((entropy_numpy, model(image).view(-1).cpu().numpy()))
            
    entropy_numpy = np.sort(entropy_numpy)
    size = entropy_numpy.shape[0]
    print(size)
    
    with open("scripts/tools/thresholds/entropy_thresholds_{}_{}_patch-{}.json".format(opt.dataset_type, opt.split, opt.patch_size), "w") as f:
        data = {}
        for i in range(99):
            cur_threshold = entropy_numpy[int((size * (i + 1)) // 100)]
            cur_threshold = cur_threshold.item()
            data["{}".format(str(i+1))] = cur_threshold
        json.dump(data, f)