import torch
import math
import torch.nn as nn
import numpy as np
from einops import rearrange

import os, sys
sys.path.append(os.getcwd())
from modules.diffusionmodules.model import ResnetBlock, AttnBlock, Upsample, Normalize, nonlinearity
from modules.dynamic_modules.tools import trunc_normal_
from modules.dynamic_modules.fourier_embedding import FourierPositionEmbedding

class PositionEmbedding2DLearned(nn.Module):
    def __init__(self, n_row, feats_dim, n_col=None):
        super().__init__()
        n_col = n_col if n_col is not None else n_row
        self.row_embed = nn.Embedding(n_row, feats_dim)
        self.col_embed = nn.Embedding(n_col, feats_dim)
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.uniform_(self.row_embed.weight)
        # nn.init.uniform_(self.col_embed.weight)
        trunc_normal_(self.row_embed.weight)
        trunc_normal_(self.col_embed.weight)

    def forward(self, x):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i).unsqueeze(0).repeat(h, 1, 1)
        y_emb = self.row_embed(j).unsqueeze(1).repeat(1, w, 1)
        pos = (x_emb + y_emb).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        
        if x.dim() == 5:  # relative position embedding
            pos = pos.unsqueeze(-3)

        x = x + pos
        return x

class Decoder(nn.Module):
    def __init__(self, 
                 ch, in_ch, out_ch, ch_mult, num_res_blocks, resolution,
                 attn_resolutions, dropout = 0.0, resamp_with_conv = True, give_pre_end = False,
                 latent_size = 32, window_size = 2, position_type = "relative"
                 ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_ch = in_ch
        self.temb_ch = 0
        self.ch = ch
        self.give_pre_end = give_pre_end

        # compute block_in and curr_res at lowest res
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,in_ch,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))

        self.conv_in = torch.nn.Conv2d(in_ch, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)
        
        # relative pos embeddings
        self.position_type = position_type
        if self.position_type == "learned":
            self.position_bias = PositionEmbedding2DLearned(n_row=latent_size, feats_dim=in_ch)
        elif self.position_type == "learned-relative":
            self.position_bias = PositionEmbedding2DLearned(n_row=window_size, feats_dim=in_ch)
            self.window_size = window_size
            self.window_num = latent_size // window_size
        elif self.position_type == "fourier":
            self.position_bias = FourierPositionEmbedding(coord_size=latent_size, hidden_size=in_ch)
        elif self.position_type == "fourier+learned":
            self.position_bias_fourier = FourierPositionEmbedding(coord_size=latent_size, hidden_size=in_ch)
            self.position_bias_learned = PositionEmbedding2DLearned(n_row=latent_size, feats_dim=in_ch)
        else:
            raise NotImplementedError()
    
    def forward(self, h, grain_indices):
        if self.position_type == "full" or self.position_type == "fourier":
            h = self.position_bias(h)
        elif self.position_type == "relative":
            h = rearrange(h, "B C (n1 nH) (n2 nW) -> B C (n1 n2) nH nW", n1=self.window_num, nH=self.window_size, n2=self.window_num, nW=self.window_size)
            h = self.position_bias(h)
            h = rearrange(h, "B C (n1 n2) nH nW -> B C (n1 nH) (n2 nW)", n1=self.window_num, nH=self.window_size)
        elif self.position_type == "fourier+learned":
            h = self.position_bias_fourier(h)
            h = self.position_bias_learned(h)

        # z to block_in
        temb = None
        h = self.conv_in(h)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h