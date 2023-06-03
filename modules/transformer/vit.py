import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import trunc_normal_


from modules.transformer.modules import PatchEmbed, Block
from modules.transformer.position_embeddings import build_position_embed

# vision transformer 
class VisionTransformerEncoder(nn.Module):
    def __init__(self, image_size, patch_size, input_channel, embed_dim, init_type, 
                 pos_embed_type, attn_drop_rate, drop_rate, depth, num_heads,
                 mlp_ratio=4, norm_type="layer", act_type="GELU", init_values=0, attn_type="sa"):
        super().__init__()

        self.patch_embed = PatchEmbed(
            img_size=image_size, patch_size=patch_size, in_chans=input_channel, embed_dim=embed_dim
        )

        self.hw = image_size // patch_size
        self.pos_emb = build_position_embed(
            embed_type=pos_embed_type, feats_dim=embed_dim, dropout=drop_rate, n_row=self.hw
        )

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate, 
                attn_drop=attn_drop_rate, init_values=init_values, act_type=act_type, 
                norm_type=norm_type, attn_type=attn_type, size=int(self.hw))
            for i in range(depth)])

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

    def forward(self, images):
        x = self.patch_embed(images)
        x = rearrange(x, "B (H W) C -> B C H W", H=self.hw, W=self.hw).contiguous()
        x = self.pos_emb(x)
        for blk in self.blocks:
            x = blk(x)
        x = rearrange(x, "B (H W) C -> B C H W", H=self.hw, W=self.hw).contiguous()
        return x


# vision transformer decoder
class VisionTransformerDecoder(nn.Module):
    def __init__(self, image_size, patch_size, pos_embed_type, embed_dim, drop_rate, 
                 depth, num_heads, attn_drop_rate, output_channel, init_type, 
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
        
        # see: VECTOR-QUANTIZED IMAGE MODELING WITH IMPROVED VQGAN
        # self.output_linear = nn.Linear(embed_dim, patch_size*patch_size*output_channel)
        # self.output_linear = nn.Sequential(
        #     nn.Linear(embed_dim, embed_dim),
        #     nn.Tanh(),
        #     nn.Linear(embed_dim, patch_size*patch_size*output_channel)
        # )
        self.output_linear1 = nn.Linear(embed_dim, patch_size*patch_size*output_channel)
        self.conv_out = nn.Linear(patch_size*patch_size*output_channel, patch_size*patch_size*output_channel)
        # align with VQGAN

        self.output_channel = output_channel
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

        # x = self.output_linear(x)
        x = self.output_linear1(x)
        x = nn.Tanh()(x)
        x = self.conv_out(x)
        x = rearrange(
            x, "B (h w) (h_size w_size c) -> B c (h h_size) (w w_size)", 
            B=B, h=self.hw, w=self.hw, h_size=self.patch_size, w_size=self.patch_size, c=self.output_channel).contiguous()
        return x

if __name__ == "__main__":
    images = torch.randn(10,3,256,256)
    vit_encoder = VisionTransformerEncoder(
        image_size=256, patch_size=16, input_channel=3, embed_dim=512, init_type="default", 
        pos_embed_type="learned-2d", attn_drop_rate=0., drop_rate=0., depth=6, num_heads=4
    )
    x = vit_encoder(images)
    print(x.size())

    vit_decoder = VisionTransformerDecoder(
        image_size=256, patch_size=16, pos_embed_type="learned-2d", embed_dim=512, drop_rate=0., 
        depth=6, num_heads=4, attn_drop_rate=0., output_channel=3, init_type="default"
    )
    y = vit_decoder(x)
    print(y.size())