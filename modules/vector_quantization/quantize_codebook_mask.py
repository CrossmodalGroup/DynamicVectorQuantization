import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
import time

import os, sys
sys.path.append(os.getcwd())

import modules.vector_quantization.common_utils as utils

# support: cosine-similarity; kmeans centroids initialization; orthogonal_reg_weight
class MaskVectorQuantize(nn.Module):
    def __init__(
        self,
        codebook_size,
        codebook_dim = None,
        kmeans_init = False,
        kmeans_iters = 10,
        use_cosine_sim = False,
        channel_last = False,
        accept_image_fmap = True,
        commitment_beta = 0.25,
        orthogonal_reg_weight = 0.,
        activate_mask_quantize = True,
    ):
        super().__init__()

        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.accept_image_fmap = accept_image_fmap
        self.channel_last = channel_last
        self.use_cosine_sim = use_cosine_sim
        self.beta = commitment_beta

        self.embedding = nn.Embedding(self.codebook_size, self.codebook_dim)

        # codebook initialization
        if not kmeans_init:
            self.embedding.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)
        else:
            self.embedding.weight.data.zero_()
        self.kmeans_iters = kmeans_iters
        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.zeros(1, codebook_size))

        self.sample_fn = utils.batched_sample_vectors
        self.all_reduce_fn = utils.noop

        # codebook_orthogonal_loss
        self.orthogonal_reg_weight = orthogonal_reg_weight

        # activate mask quantization
        self.activate_mask_quantize = activate_mask_quantize
    
    def init_embed_(self, data):
        if self.initted:
            return
        
        data = rearrange(data, '... -> 1 ...').contiguous()
        data = rearrange(data, 'h ... d -> h (...) d').contiguous()

        embed, cluster_size = utils.kmeans(
            data,
            self.codebook_size,
            self.kmeans_iters,
            sample_fn = self.sample_fn,
            all_reduce_fn = self.all_reduce_fn
        )

        self.embedding.weight.data.copy_(embed.squeeze(0))
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))
    
    def forward(self, x, temp=0., codebook_mask=None):
        need_transpose = not self.channel_last and not self.accept_image_fmap

        if self.accept_image_fmap:
            height, width = x.shape[-2:]
            x = rearrange(x, 'b c h w -> b (h w) c').contiguous()

            if codebook_mask is not None and self.activate_mask_quantize:
                codebook_mask = rearrange(codebook_mask, "b c h w -> b (h w) c").contiguous()

        if need_transpose:
            x = rearrange(x, 'b d n -> b n d').contiguous()
        shape, device, dtype = x.shape, x.device, x.dtype
        flatten = rearrange(x, 'h ... d -> h (...) d').contiguous()

        # if use cosine_sim, whether should norm the feature before k-means initialization ?
        # if self.use_cosine_sim:
        #     flatten = F.normalize(flatten, p = 2, dim = -1)
        self.init_embed_(flatten)

        # calculate the distance
        if self.use_cosine_sim:  # cosine similarity
            flatten_norm = F.normalize(flatten, p = 2, dim = -1)
            weight_norm = F.normalize(self.embedding.weight, p = 2, dim = -1).unsqueeze(0)
            
            # compute inner product
            dist = einsum('h n d, h c d -> h n c', flatten_norm, weight_norm)
        else:  # L2 distance 
            flatten = flatten.view(-1, self.codebook_dim)
            dist = - torch.sum(flatten ** 2, dim=1, keepdim=True) - \
                torch.sum(self.embedding.weight**2, dim=1) + 2 * \
                torch.einsum('bd,dn->bn', flatten, rearrange(self.embedding.weight, 'n d -> d n'))  # more efficient, add "-" for argmax gumbel sample

        embed_ind = utils.gumbel_sample(dist, dim = -1, temperature = temp)
        # embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = embed_ind.view(*shape[:-1])
        
        x_q = self.embedding(embed_ind)

        # compute loss for embedding
        if codebook_mask is not None and self.activate_mask_quantize:
            ratio = 1 / torch.mean(codebook_mask)
            loss = ratio * self.beta * torch.mean((x_q.detach()-x) ** 2 * codebook_mask) + ratio * torch.mean((x_q - x.detach()) ** 2 * codebook_mask)
        else:
            loss = self.beta * torch.mean((x_q.detach()-x)**2) + torch.mean((x_q - x.detach()) ** 2)

        # ortho reg term
        if self.orthogonal_reg_weight > 0. :
            # eq (2) from https://arxiv.org/abs/2112.00384
            emb_weight_after_norm = F.normalize(self.embedding.weight, p = 2, dim = -1)
            diff = torch.mm(emb_weight_after_norm, torch.transpose(emb_weight_after_norm, 0, 1)) - torch.eye(self.codebook_size, self.codebook_size).type_as(emb_weight_after_norm)
            ortho_reg_term = self.orthogonal_reg_weight * torch.sum(diff**2) / (diff.size(0)**2)

            # diff = torch.mm(self.embedding.weight, torch.transpose(self.embedding.weight, 0, 1)) - torch.eye(self.codebook_size, self.codebook_size).type_as(self.embedding.weight)
            # ortho_reg_term = self.orthogonal_reg_weight * torch.sum(diff**2) / (diff.size(0)**2)
            loss = loss + ortho_reg_term

        # preserve gradients
        x_q = x + (x_q - x).detach()

        if need_transpose:
            x_q = rearrange(x_q, 'b n d -> b d n').contiguous()

        if self.accept_image_fmap:
            x_q = rearrange(x_q, 'b (h w) c -> b c h w', h = height, w = width).contiguous()
            embed_ind = rearrange(embed_ind, 'b (h w) ... -> b h w ...', h = height, w = width).contiguous()

        return x_q, loss, (None, None, embed_ind)
    
    def get_codebook_entry(self, indices, shape, *kwargs):
        # get quantized latent vectors
        z_q = self.embedding(indices)  # (batch, height, width, channel)
        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q
    
    @torch.no_grad()
    def embed_code_with_depth(self, code, to_latent_shape=False):        
        code_slices = torch.chunk(code, chunks=code.shape[-1], dim=-1)

        embeds = [self.embedding(code_slice) for i, code_slice in enumerate(code_slices)]

        if to_latent_shape:
            embeds = [self.to_latent_shape(embed.squeeze(-2)).unsqueeze(-2) for embed in embeds]
        embeds = torch.cat(embeds, dim=-2)
        
        return embeds, None

if __name__ == "__main__":
    # quantizer = VectorQuantize(
    #     codebook_size = 1024,
    #     codebook_dim = 512,
    #     kmeans_init = True,
    #     kmeans_iters = 10,
    #     use_cosine_sim = False,
    #     channel_last = False,
    #     accept_image_fmap = False,
    #     commitment_beta = 0.25,
    #     orthogonal_reg_weight = 10.,
    #     use_ddp = False,
    # )

    # # x = torch.randn(10, 512, 16, 16)
    # x = torch.randn(10, 512, 120)

    # x_q, loss, (_, _, embed_ind) = quantizer(x, 0.)
    # print(loss)
    pass