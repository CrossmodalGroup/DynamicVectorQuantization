import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import trunc_normal_

from modules.transformer.modules import Block
from modules.transformer.position_embeddings import build_position_embed
from utils.utils import instantiate_from_config

class VisionTransformerDecoder(nn.Module):
    def __init__(self, image_size, patch_size, pos_embed_type, embed_dim,  
                 depth, num_heads, attn_drop_rate=0., drop_rate=0.,init_type="default", 
                 mlp_ratio=4, act_type="GELU", norm_type="layer", attn_type="sa", init_values=0):
        super().__init__()

        self.hw = image_size // patch_size
        self.pos_emb = build_position_embed(embed_type=pos_embed_type, feats_dim=embed_dim, dropout=drop_rate, n_row=self.hw)
        self.blocks = nn.ModuleList([
            Block(
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
    
    def forward(self, x):
        B = x.size(0)
        x = self.pos_emb(x)

        for blk in self.blocks:
            x = blk(x)
        
        x = rearrange(x, "B (H W) C -> B C H W", H=self.hw, W=self.hw).contiguous()
        return x


class HybrdDecoder(nn.Module):
    def __init__(self, transformer_config, cnn_config):
        super().__init__()
        self.transformer = instantiate_from_config(transformer_config)
        self.cnn = instantiate_from_config(cnn_config)
        self.conv_out = self.cnn.conv_out
    
    def forward(self, x):
        x = self.transformer(x)
        x = self.cnn(x)
        return x
    
# transformer 模块的输入还加入了mask，以实现自定义的功能
class HybrdDecoder_V2(nn.Module):
    def __init__(self, transformer_config, cnn_config):
        super().__init__()
        self.transformer = instantiate_from_config(transformer_config)
        self.cnn = instantiate_from_config(cnn_config)
        self.conv_out = self.cnn.conv_out
    
    def forward(self, x, mask):
        x = self.transformer(x, mask)
        x = self.cnn(x)
        return x

if __name__ == "__main__":  
    x = torch.randn(10, 512, 16, 16)
    vit_decoder = VisionTransformerDecoder(
        image_size=256, patch_size=16, pos_embed_type="learned-2d", embed_dim=512, drop_rate=0., 
        depth=6, num_heads=4, attn_drop_rate=0., output_channel=3, init_type="default"
    )
    y = vit_decoder(x)
    print(y.size())