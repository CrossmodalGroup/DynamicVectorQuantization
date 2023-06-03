import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import trunc_normal_

import os, sys
sys.path.append(os.getcwd())

from modules.transformer.modules import norm, Mlp
from modules.transformer.mask_attention import MaskSelfAttention_SquareGrowth
from modules.transformer.position_embeddings import build_position_embed

class MaskBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0., init_values=None, 
                 act_type="GELU", norm_type="layer",  attn_type="msa-sg", size=None):
        super().__init__()
        self.norm1 = norm(norm_type, dim)
        
        self.norm2 = norm(norm_type, dim)
        self.mlp = Mlp(dim, hidden_radio=mlp_ratio, act_type=act_type, drop=drop)
        
        if attn_type == "msa-sg":
            self.attn = MaskSelfAttention_SquareGrowth(
                dim=dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop
            )
        else:
            raise ValueError

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None
    
    def forward(self, x, mask, **ignore_kwargs):
        if self.gamma_1 is None:
            attn, new_mask = self.attn(h=self.norm1(x), mask=mask)
            x = x + attn
            x = x + self.mlp(self.norm2(x))
        else:
            attn, new_mask = self.attn(h=self.norm1(x), mask=mask)
            x = x + self.gamma_1 * attn
            x = x + self.gamma_2 * self.mlp(self.norm2(x))
        return x, new_mask

class MaskVisionTransformerDecoder(nn.Module):
    def __init__(self, image_size, patch_size, pos_embed_type, embed_dim,  
                 depth, num_heads, attn_drop_rate=0., drop_rate=0.,init_type="default", 
                 mlp_ratio=4, act_type="GELU", norm_type="layer", attn_type="sa", init_values=0):
        super().__init__()

        self.hw = image_size // patch_size
        self.pos_emb = build_position_embed(embed_type=pos_embed_type, feats_dim=embed_dim, dropout=drop_rate, n_row=self.hw)
        self.blocks = nn.ModuleList([
            MaskBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate, 
                attn_drop=attn_drop_rate, init_values=init_values, act_type=act_type, 
                norm_type=norm_type, attn_type=attn_type, size=int(self.hw))
            for i in range(depth)])

        self.patch_size = patch_size
        
        if init_type == "default":
            self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, mask):
        B = x.size(0)
        x = self.pos_emb(x)

        new_mask = mask + 0.02 * (1 - mask) # 将0初始化为一个小值，0.02
        for blk in self.blocks:
            # print(new_mask)
            x, new_mask = blk(x=x, mask=new_mask)
        x = rearrange(x, "B (H W) C -> B C H W", H=self.hw, W=self.hw).contiguous()
        return x