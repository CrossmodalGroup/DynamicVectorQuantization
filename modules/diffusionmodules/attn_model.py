import torch
import torch.nn as nn
import numpy as np

import os, sys
sys.path.append(os.getcwd())

from modules.diffusionmodules.model import ResnetBlock, AttnBlock, Upsample, Normalize, nonlinearity

# resnet block with chosen kernel;
# for kernel size, 1 is not good, 3 is good
class ResnetBlock_kernel_1(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512, kernel_size=1):
        super().__init__()
        if kernel_size == 3:
            padding = 1
        elif kernel_size == 1:
            padding = 0
            
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=padding)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=padding)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=kernel_size,
                                                     stride=1,
                                                     padding=padding)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb=None, **ignore_kwargs):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

class SelfAttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x, **ignore_kwargs):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_

class BiasedSelfAttnBlock(nn.Module):
    def __init__(self, in_channels, reweight=False):
        super().__init__()
        self.in_channels = in_channels
        self.apply_reweight = reweight
        
        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, mask, **ignore_kwargs):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)
        
        if mask is not None:
            unsqueezed_mask = mask.unsqueeze(-2)
            w_ = w_ * unsqueezed_mask
            
            if self.apply_reweight:
                w_sum = torch.sum(w_, dim=-1, keepdim=True)
                w_ = w_ / w_sum

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_

    
class TokenReconstruction(nn.Module):
    # structure: [resnet, {attn, resnet} * n_layer]
    def __init__(self, n_layer, input_dim, dropout, attn_type="self-attn", 
                 resnet_kernel_size=1, mask_update_mode="square", reweight=False,
                 fix_bug=False):
        super().__init__()
        self.n_layer = n_layer
        self.mask_update_mode = mask_update_mode
        self.middle = nn.ModuleList()
        self.middle.append(ResnetBlock_kernel_1(
            in_channels=input_dim, dropout=dropout, kernel_size=resnet_kernel_size
        ))
        for i in range(self.n_layer):
            if attn_type == "self-attn":
                self.middle.append(
                    SelfAttnBlock(input_dim)
                )
            elif attn_type == "bias-self-attn":
                self.middle.append(
                    BiasedSelfAttnBlock(input_dim, reweight)
                )
            else:
                raise ValueError()
            if not fix_bug:
                self.middle.append(
                    ResnetBlock_kernel_1(in_channels=input_dim, dropout=dropout),
                )
            else:
                self.middle.append(
                    ResnetBlock_kernel_1(in_channels=input_dim, dropout=dropout, kernel_size=resnet_kernel_size),
                )

    def forward(self, x, mask):
        if self.mask_update_mode == "square" or self.mask_update_mode == "cube":
            mask = mask + 0.02 * (1 - mask) # 将0初始化为一个小值，0.02
        elif self.mask_update_mode == "linear":
            gain = 1 / (self.n_layer-1)
            original_mask = mask
        # elif self.mask_update_mode == "const":
        #     pass

        i = 0
        for module in self.middle:  # [resnet, attn, resnet, attn, ...]
            x = module(x=x, mask=mask)
            
            if i % 2 == 1: # pass attn layer then update mask
                # update mask
                if self.mask_update_mode == "const":
                    mask = mask
                elif self.mask_update_mode == "square":
                    mask = torch.sqrt(mask)
                elif self.mask_update_mode == "cube":
                    mask = torch.pow(mask, (1/3))
                elif self.mask_update_mode == "linear":
                    mask = original_mask + (i//2 + 1) * gain * (1 - original_mask)
                else:
                    raise ValueError
            i += 1
        return x
        

# replace original middle layer with TokenReconstruction module
class AttnDecoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, 
                 token_n_layer=6, token_attn_type="self-attn", 
                 resnet_kernel_size=1, mask_update_mode="square", reweight=False,
                 fix_bug=False,
                 **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        
        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))
        
        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        
        # middle
        self.mid = TokenReconstruction(
            n_layer=token_n_layer, input_dim=block_in, dropout=dropout, 
            attn_type=token_attn_type, resnet_kernel_size=resnet_kernel_size,
            mask_update_mode=mask_update_mode, reweight=reweight, fix_bug=fix_bug
        )
        
        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
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
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
    
    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.ones(x.size(0), x.size(2) * x.size(3)).to(x.device)
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = x.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(x)

        # middle
        h = self.mid(h, mask)
            
        
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
    

if __name__ == "__main__":
    # resnet_block = ResnetBlock_kernel_1(
    #     in_channels=256, out_channels=512, conv_shortcut=True, dropout=0.
    # )
    
    x = torch.randn(10,256,16,16)
    mask = torch.randint(0,2,(10,256))
    # y, _ = resnet_block(x, None)
    
    # print(y.size())
    
    # bias_attn = BiasedSelfAttnBlock(in_channels=256, reweight=True)
    # y = bias_attn(x, mask)
    # exit()
    
    model = TokenReconstruction(
        n_layer=6, input_dim=256, dropout=0., attn_type="bias-self-attn", 
        mask_update_mode="cube", reweight=True
    )
    
    y = model(x=x, mask=mask)
    print(y.size())
    
    model = AttnDecoder(
        ch=32, out_ch=3, ch_mult=(1,1,2,2,4), num_res_blocks=1,
        attn_resolutions=[], dropout=0.0, resamp_with_conv=True, in_channels=256,
        resolution=256, z_channels=256, give_pre_end=False, 
        token_n_layer=6, token_attn_type="bias-self-attn", mask_update_mode="square"
    )
    
    y = model(x=x, mask=mask)
    print(y.size())