import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import einsum

# fourier

class SinusoidalPosEmb(nn.Module):
    def __init__(
        self,
        dim,
        height_or_width,
        theta = 10000
    ):
        super().__init__()
        self.dim = dim
        self.theta = theta

        hw_range = torch.arange(height_or_width)
        # coors = torch.stack(torch.meshgrid(hw_range, hw_range, indexing = 'ij'), dim = -1)
        coors = torch.stack(torch.meshgrid(hw_range, hw_range), dim = -1)
        coors = rearrange(coors, 'h w c -> h w c')
        self.register_buffer('coors', coors, persistent = False)

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device = x.device) * -emb)
        emb = rearrange(self.coors, 'h w c -> h w c 1') * rearrange(emb, 'j -> 1 1 1 j')
        fourier = torch.cat((emb.sin(), emb.cos()), dim = -1)
        fourier = repeat(fourier, 'h w c d -> b (c d) h w', b = x.shape[0])
        return torch.cat((x, fourier), dim = 1)

# 2d relative positional bias

class RelPosBias2d(nn.Module):
    def __init__(self, size, heads):
        super().__init__()
        self.pos_bias = nn.Embedding((2 * size - 1) ** 2, heads)

        arange = torch.arange(size)

        # pos = torch.stack(torch.meshgrid(arange, arange, indexing = 'ij'), dim = -1)
        pos = torch.stack(torch.meshgrid(arange, arange), dim = -1)
        pos = rearrange(pos, '... c -> (...) c')
        rel_pos = rearrange(pos, 'i c -> i 1 c') - rearrange(pos, 'j c -> 1 j c')

        rel_pos = rel_pos + size - 1
        h_rel, w_rel = rel_pos.unbind(dim = -1)
        pos_indices = h_rel * (2 * size - 1) + w_rel
        self.register_buffer('pos_indices', pos_indices)

    def forward(self, qk):
        i, j = qk.shape[-2:]

        bias = self.pos_bias(self.pos_indices)
        bias = rearrange(bias, 'i j h -> h i j')
        return bias

# ViT encoder / decoder

class ChanLayerNorm(nn.Module):
    def __init__(
        self,
        dim,
        eps = 1e-5
    ):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + self.eps).rsqrt() * self.gamma

class PEG(nn.Module):
    def __init__(self, dim, kernel_size = 3):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, kernel_size = kernel_size, padding = kernel_size // 2, groups = dim, stride = 1)

    def forward(self, x):
        return self.proj(x)

class SPT(nn.Module):
    """ https://arxiv.org/abs/2112.13492 """

    def __init__(self, *, dim, patch_size, channels = 3):
        super().__init__()
        patch_dim = patch_size * patch_size * 5 * channels

        self.to_patch_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w', p1 = patch_size, p2 = patch_size),
            ChanLayerNorm(patch_dim),
            nn.Conv2d(patch_dim, dim, 1)
        )

    def forward(self, x):
        shifts = ((1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1))
        shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))
        x_with_shifts = torch.cat((x, *shifted_x), dim = 1)
        return self.to_patch_tokens(x_with_shifts)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        heads = 8,
        dim_head = 32,
        fmap_size = None,
        rel_pos_bias = False
    ):
        super().__init__()
        self.norm = ChanLayerNorm(dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias = False)
        self.primer_ds_convs = nn.ModuleList([PEG(inner_dim) for _ in range(3)])

        self.to_out = nn.Conv2d(inner_dim, dim, 1, bias = False)

        self.rel_pos_bias = None
        if rel_pos_bias:
            assert fmap_size is not None
            self.rel_pos_bias = RelPosBias2d(fmap_size, heads)

    def forward(self, x):
        fmap_size = x.shape[-1]
        h = self.heads

        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = 1)

        q, k, v = [ds_conv(t) for ds_conv, t in zip(self.primer_ds_convs, (q, k, v))]
        # q, k, v = rearrange_many((q, k, v), 'b (h d) x y -> b h (x y) d', h = h)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h = h), (q, k, v))

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        if self.rel_pos_bias is not None:
            sim = sim + self.rel_pos_bias(sim)

        attn = sim.softmax(dim = -1, dtype = torch.float32)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = fmap_size, y = fmap_size)
        return self.to_out(out)

def FeedForward(dim, mult = 4):
    return nn.Sequential(
        ChanLayerNorm(dim),
        nn.Conv2d(dim, dim * mult, 1, bias = False),
        nn.GELU(),
        PEG(dim * mult),
        nn.Conv2d(dim * mult, dim, 1, bias = False)
    )

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        layers,
        dim_head = 32,
        heads = 8,
        ff_mult = 4,
        fmap_size = None
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(layers):
            self.layers.append(nn.ModuleList([
                PEG(dim = dim),
                Attention(dim = dim, dim_head = dim_head, heads = heads, fmap_size = fmap_size, rel_pos_bias = True),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.norm = ChanLayerNorm(dim)

    def forward(self, x):
        for peg, attn, ff in self.layers:
            x = peg(x) + x
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViTEncoder(nn.Module):
    def __init__(
        self,
        dim,
        image_size,
        channels = 3,
        layers = 4,
        patch_size = 16,
        dim_head = 32,
        heads = 8,
        ff_mult = 4
    ):
        super().__init__()
        self.encoded_dim = dim
        self.patch_size = patch_size

        fmap_size = image_size // patch_size

        self.encoder = nn.Sequential(
            SPT(dim = dim, patch_size = patch_size, channels = channels),
            Transformer(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                ff_mult = ff_mult,
                layers = layers,
                fmap_size = fmap_size
            ),
        )
    def forward(self, x):
        return self.encoder(x)


class ViTDecoder(nn.Module):
    def __init__(
        self,
        dim,
        image_size,
        channels = 3,
        layers = 4,
        patch_size = 16,
        dim_head = 32,
        heads = 8,
        ff_mult = 4
    ):
        super().__init__()
        self.encoded_dim = dim
        self.patch_size = patch_size

        input_dim = channels * (patch_size ** 2)
        fmap_size = image_size // patch_size

        self.decoder = nn.Sequential(
            Transformer(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                ff_mult = ff_mult,
                layers = layers,
                fmap_size = fmap_size
            ),
            nn.Sequential(
                SinusoidalPosEmb(dim // 2, height_or_width = fmap_size),
                nn.Conv2d(2 * dim, dim * 4, 3, bias = False, padding = 1),
                nn.Tanh(),
                nn.Conv2d(dim * 4, input_dim, 1, bias = False),
            ),
            Rearrange('b (p1 p2 c) h w -> b c (h p1) (w p2)', p1 = patch_size, p2 = patch_size)
        )

    @property
    def last_layer(self):
        return self.decoder[-2][-1].weight

    def forward(self, x):
        return self.decoder(x)

if __name__ == "__main__":
    x = torch.randn(10, 3, 256, 256)
    encoder = ViTEncoder(
        dim = 512,
        image_size = 256,
        channels = 3,
        layers = 4,
        patch_size = 32,
        dim_head = 32,
        heads = 8,
        ff_mult = 4
    )

    out = encoder(x)
    print(out.size())

    decoder = ViTDecoder(
        dim = 512,
        image_size = 256,
        channels = 3,
        layers = 4,
        patch_size = 32,
        dim_head = 32,
        heads = 8,
        ff_mult = 4
    )

    x_rec = decoder(out)
    print(x_rec.size())