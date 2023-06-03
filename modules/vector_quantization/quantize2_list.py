# add the random restart of rqvae
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from einops import rearrange


class VQEmbedding(nn.Embedding):
    """VQ embedding module with ema update."""

    def __init__(self, n_embed, embed_dim, ema=True, decay=0.99, restart_unused_codes=True, eps=1e-5):
        super().__init__(n_embed + 1, embed_dim, padding_idx=n_embed)

        self.ema = ema
        self.decay = decay
        self.eps = eps
        self.restart_unused_codes = restart_unused_codes
        self.n_embed = n_embed

        if self.ema:
            _ = [p.requires_grad_(False) for p in self.parameters()]

            # padding index is not updated by EMA
            self.register_buffer('cluster_size_ema', torch.zeros(n_embed))
            self.register_buffer('embed_ema', self.weight[:-1, :].detach().clone())

    @torch.no_grad()
    def compute_distances(self, inputs):
        codebook_t = self.weight[:-1, :].t()

        (embed_dim, _) = codebook_t.shape
        inputs_shape = inputs.shape
        assert inputs_shape[-1] == embed_dim

        inputs_flat = inputs.reshape(-1, embed_dim)

        inputs_norm_sq = inputs_flat.pow(2.).sum(dim=1, keepdim=True)
        codebook_t_norm_sq = codebook_t.pow(2.).sum(dim=0, keepdim=True)
        distances = torch.addmm(
            inputs_norm_sq + codebook_t_norm_sq,
            inputs_flat,
            codebook_t,
            alpha=-2.0,
        )
        distances = distances.reshape(*inputs_shape[:-1], -1)  # [B, h, w, n_embed or n_embed+1]
        return distances

    @torch.no_grad()
    def find_nearest_embedding(self, inputs):
        distances = self.compute_distances(inputs)  # [B, h, w, n_embed or n_embed+1]
        embed_idxs = distances.argmin(dim=-1)  # use padding index or not

        return embed_idxs

    @torch.no_grad()
    def _tile_with_noise(self, x, target_n):
        B, embed_dim = x.shape
        n_repeats = (target_n + B -1) // B
        std = x.new_ones(embed_dim) * 0.01 / np.sqrt(embed_dim)
        x = x.repeat(n_repeats, 1)
        x = x + torch.rand_like(x) * std
        return x    
    
    @torch.no_grad()
    def _update_buffers(self, vectors, idxs):

        n_embed, embed_dim = self.weight.shape[0]-1, self.weight.shape[-1]
        
        vectors = vectors.reshape(-1, embed_dim)
        idxs = idxs.reshape(-1)
        
        n_vectors = vectors.shape[0]
        n_total_embed = n_embed

        one_hot_idxs = vectors.new_zeros(n_total_embed, n_vectors)
        one_hot_idxs.scatter_(dim=0,
                              index=idxs.unsqueeze(0),
                              src=vectors.new_ones(1, n_vectors)
                              )

        cluster_size = one_hot_idxs.sum(dim=1)
        vectors_sum_per_cluster = one_hot_idxs @ vectors

        if dist.is_initialized():
            dist.all_reduce(vectors_sum_per_cluster, op=dist.ReduceOp.SUM)
            dist.all_reduce(cluster_size, op=dist.ReduceOp.SUM)

        self.cluster_size_ema.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        self.embed_ema.mul_(self.decay).add_(vectors_sum_per_cluster, alpha=1 - self.decay)
        
        if self.restart_unused_codes:
            if n_vectors < n_embed:
                vectors = self._tile_with_noise(vectors, n_embed)
            n_vectors = vectors.shape[0]
            _vectors_random = vectors[torch.randperm(n_vectors, device=vectors.device)][:n_embed]
            
            if dist.is_initialized():
                dist.broadcast(_vectors_random, 0)
        
            usage = (self.cluster_size_ema.view(-1, 1) >= 1).float()
            self.embed_ema.mul_(usage).add_(_vectors_random * (1-usage))
            self.cluster_size_ema.mul_(usage.view(-1))
            self.cluster_size_ema.add_(torch.ones_like(self.cluster_size_ema) * (1-usage).view(-1))

    @torch.no_grad()
    def _update_embedding(self):

        n_embed = self.weight.shape[0] - 1
        n = self.cluster_size_ema.sum()
        normalized_cluster_size = (
            n * (self.cluster_size_ema + self.eps) / (n + n_embed * self.eps)
        )
        self.weight[:-1, :] = self.embed_ema / normalized_cluster_size.reshape(-1, 1)

    def forward(self, inputs):
        embed_idxs = self.find_nearest_embedding(inputs)
        if self.training:
            if self.ema:
                self._update_buffers(inputs, embed_idxs)
        
        embeds = self.embed(embed_idxs)

        if self.ema and self.training:
            self._update_embedding()

        return embeds, embed_idxs

    def embed(self, idxs):
        embeds = super().forward(idxs)
        return embeds

# simplified version with random restart unused, accept list features input
class VectorQuantize2(nn.Module):
    def __init__(self,
                codebook_size,
                codebook_dim = None,
                commitment_beta = 0.25,
                decay = 0.99,
                restart_unused_codes = True,
                ):
        super().__init__()
        self.beta = commitment_beta
        self.restart_unused_codes = restart_unused_codes
        self.codebook = VQEmbedding(codebook_size, 
                                    codebook_dim, 
                                    decay = decay, 
                                    restart_unused_codes = restart_unused_codes,
                                    )
        self.codebook.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
    
    def forward(self, x_list, *ignorewargs, **ignorekwargs):
        batch_size = len(x_list)
        
        x_q_list, x_code_list = [], []
        loss = 0.
        for i in range(batch_size):
            x_q_i, x_code_i = self.codebook(x_list[i])
            
            loss += self.beta * torch.mean((x_q_i.detach()-x_list[i])**2) + torch.mean((x_q_i - x_list[i].detach()) ** 2)
            
            # preserve gradients
            x_q_i = x_list[i] + (x_q_i - x_list[i]).detach()
            
            x_q_list.append(x_q_i)
            x_code_list.append(x_code_i)

        loss /= batch_size
        return x_q_list, loss, (None, None, x_code_list)
    
    @torch.no_grad()
    def get_soft_codes(self, x, temp=1.0, stochastic=False):
        distances = self.codebook.compute_distances(x)
        soft_code = F.softmax(-distances / temp, dim=-1)

        if stochastic:
            soft_code_flat = soft_code.reshape(-1, soft_code.shape[-1])
            code = torch.multinomial(soft_code_flat, 1)
            code = code.reshape(*soft_code.shape[:-1])
        else:
            code = distances.argmin(dim=-1)

        return soft_code, code
    
    def get_codebook_entry(self, indices, *kwargs):
        # get quantized latent vectors
        z_q = self.codebook.embed(indices)  # (batch, height, width, channel)
        return z_q