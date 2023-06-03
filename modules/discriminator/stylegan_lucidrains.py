import torch
import torch.nn as nn

def default(val, d):
    return val if val is not None else d

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

class CrossEmbedLayer(nn.Module):
    def __init__(
        self,
        dim_in,
        kernel_sizes,
        dim_out = None,
        stride = 2
    ):
        super().__init__()
        assert all([*map(lambda t: (t % 2) == (stride % 2), kernel_sizes)])
        dim_out = default(dim_out, dim_in)

        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        # calculate the dimension at each scale
        dim_scales = [int(dim_out / (2 ** i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        self.convs = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(nn.Conv2d(dim_in, dim_scale, kernel, stride = stride, padding = (kernel - stride) // 2))

    def forward(self, x):
        fmaps = tuple(map(lambda conv: conv(x), self.convs))
        return torch.cat(fmaps, dim = 1)

class Block(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        groups = 8
    ):
        super().__init__()
        self.groupnorm = nn.GroupNorm(groups, dim)
        self.activation = nn.LeakyReLU(0.1)
        self.project = nn.Conv2d(dim, dim_out, 3, padding = 1)

    def forward(self, x, scale_shift = None):
        x = self.groupnorm(x)
        x = self.activation(x)
        return self.project(x)

class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        *,
        groups = 8
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.block = Block(dim, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block(x)
        return h + self.res_conv(x)

# discriminator

class Discriminator(nn.Module):
    def __init__(
        self,
        dim = 256, 
        discr_layers = 6,
        channels = 3,
        groups = 8,
        cross_embed_kernel_sizes = (3, 7, 15)
    ):
        super().__init__()

        layer_mults = list(map(lambda t: 2 ** t, range(discr_layers)))
        layer_dims = [dim * mult for mult in layer_mults]
        dims = (dim, *layer_dims)

        init_dim, *_, final_dim = dims
        dim_pairs = zip(dims[:-1], dims[1:])

        self.layers = nn.ModuleList([nn.Sequential(
            CrossEmbedLayer(channels, cross_embed_kernel_sizes, init_dim, stride = 1),
            nn.LeakyReLU(0.1)
        )])

        for dim_in, dim_out in dim_pairs:
            self.layers.append(nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 4, stride = 2, padding = 1),
                nn.LeakyReLU(0.1),
                nn.GroupNorm(groups, dim_out),
                ResnetBlock(dim_out, dim_out),
            ))

        self.to_logits = nn.Sequential( # return 5 x 5, for PatchGAN-esque training
            nn.Conv2d(final_dim, final_dim, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(final_dim, 1, 4)
        )

    def forward(self, x):
        for net in self.layers:
            x = net(x)

        return self.to_logits(x)

if __name__ == "__main__":
    x = torch.randn(10, 3, 256, 256)

    D = Discriminator(
        dim = 256, 
        discr_layers = 6,
        channels = 3,
        groups = 8,
        cross_embed_kernel_sizes = (3, 7, 15)
    )

    y = D(x)
    print(y.size())