import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.getcwd())

import pytorch_lightning as pl
from modules.diffusionmodules.model import (AttnBlock, Downsample, Normalize, ResnetBlock, nonlinearity)
from utils.utils import instantiate_from_config

# the last two grains
class DualGrainEncoder(pl.LightningModule):
    def __init__(self, 
        *, 
        ch, 
        ch_mult=(1,2,4,8), 
        num_res_blocks,
        attn_resolutions, 
        dropout=0.0, 
        resamp_with_conv=True, 
        in_channels,
        resolution, 
        z_channels, 
        router_config=None,
        update_router=True,
        **ignore_kwargs
        ):
        super().__init__()
        
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle for the coarse grain
        self.mid_coarse = nn.Module()
        self.mid_coarse.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid_coarse.attn_1 = AttnBlock(block_in)
        self.mid_coarse.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)

        # end for the coarse grain
        self.norm_out_coarse = Normalize(block_in)
        self.conv_out_coarse = torch.nn.Conv2d(block_in, z_channels, kernel_size=3, stride=1, padding=1)
        
        block_in_finegrain = block_in // (ch_mult[-1] // ch_mult[-2])
        # middle for the fine grain
        self.mid_fine = nn.Module()
        self.mid_fine.block_1 = ResnetBlock(in_channels=block_in_finegrain, out_channels=block_in_finegrain, temb_channels=self.temb_ch, dropout=dropout)
        self.mid_fine.attn_1 = AttnBlock(block_in_finegrain)
        self.mid_fine.block_2 = ResnetBlock(in_channels=block_in_finegrain, out_channels=block_in_finegrain, temb_channels=self.temb_ch, dropout=dropout)

        # end for the fine grain
        self.norm_out_fine = Normalize(block_in_finegrain)
        self.conv_out_fine = torch.nn.Conv2d(block_in_finegrain, z_channels, kernel_size=3, stride=1, padding=1)

        self.router = instantiate_from_config(router_config)
        self.update_router = update_router


    def forward(self, x, x_entropy):
        assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))
            if i_level == self.num_resolutions-2:
                h_fine = h

        h_coarse = hs[-1]

        # middle for the h_coarse
        h_coarse = self.mid_coarse.block_1(h_coarse, temb)
        h_coarse = self.mid_coarse.attn_1(h_coarse)
        h_coarse = self.mid_coarse.block_2(h_coarse, temb)

        # end for the h_coarse
        h_coarse = self.norm_out_coarse(h_coarse)
        h_coarse = nonlinearity(h_coarse)
        h_coarse = self.conv_out_coarse(h_coarse)

        # middle for the h_fine
        h_fine = self.mid_fine.block_1(h_fine, temb)
        h_fine = self.mid_fine.attn_1(h_fine)
        h_fine = self.mid_fine.block_2(h_fine, temb)

        # end for the h_coarse
        h_fine = self.norm_out_fine(h_fine)
        h_fine = nonlinearity(h_fine)
        h_fine = self.conv_out_fine(h_fine)

        # dynamic routing
        gate = self.router(h_fine=h_fine, h_coarse=h_coarse, entropy=x_entropy)
        if self.update_router and self.training:
            gate = F.gumbel_softmax(gate, dim=-1, hard=True)
        gate = gate.permute(0,3,1,2)
        indices = gate.argmax(dim=1)

        h_coarse = h_coarse.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
        indices_repeat = indices.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2).unsqueeze(1)
        # 0 for coarse-grained and 1 for fine-grained
        h_dual = torch.where(indices_repeat==0, h_coarse, h_fine)

        if self.update_router and self.training:
            gate_grad = gate.max(dim=1, keepdim=True)[0]
            gate_grad = gate_grad.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
            h_dual = h_dual * gate_grad
        
        coarse_mask = 0.25 * torch.ones_like(indices_repeat).to(h_dual.device)
        fine_mask = 1.0 * torch.ones_like(indices_repeat).to(h_dual.device)
        codebook_mask = torch.where(indices_repeat==0, coarse_mask, fine_mask)

        return {
            "h_dual": h_dual,
            "indices": indices,
            "codebook_mask": codebook_mask,
            "gate": gate,
        }