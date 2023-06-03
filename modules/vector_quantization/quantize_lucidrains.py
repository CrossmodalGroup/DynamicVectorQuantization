# a simplified version of lucidrains/vector-quantize-pytorch

import os
import sys

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum, nn
from torch.cuda.amp import autocast

sys.path.append(os.getcwd())

import modules.vector_quantization.common_utils as utils

# regularization losses

def orthogonal_loss_fn(t):
    # eq (2) from https://arxiv.org/abs/2112.00384
    h, n = t.shape[:2]
    normed_codes = F.normalize(t, p = 2, dim = -1)
    identity = repeat(torch.eye(n, device = t.device), 'i j -> h i j', h = h)
    cosine_sim = einsum('h i d, h j d -> h i j', normed_codes, normed_codes)
    return ((cosine_sim - identity) ** 2).sum() / (h * n ** 2)

# distance types

class EuclideanCodebook(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        kmeans_init = False,
        kmeans_iters = 10,
        decay = 0.8,
        eps = 1e-5,
        threshold_ema_dead_code = 2,
        learnable_codebook = False,
        sample_codebook_temp = 0,
        use_ddp = False,
    ):
        super().__init__()
        self.decay = decay
        init_fn = utils.uniform_init if not kmeans_init else torch.zeros
        embed = init_fn(1, codebook_size, dim)

        self.codebook_size = codebook_size

        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.sample_codebook_temp = sample_codebook_temp

        self.sample_fn = utils.sample_vectors_distributed if use_ddp else utils.batched_sample_vectors
        self.all_reduce_fn = utils.distributed.all_reduce if use_ddp else utils.noop

        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.zeros(1, codebook_size))
        self.register_buffer('embed_avg', embed.clone())

        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer('embed', embed)

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.initted:
            return

        embed, cluster_size = utils.kmeans(
            data,
            self.codebook_size,
            self.kmeans_iters,
            sample_fn = self.sample_fn,
            all_reduce_fn = self.all_reduce_fn
        )

        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    def replace(self, batch_samples, batch_mask):
        batch_samples = F.normalize(batch_samples, p = 2, dim = -1)

        for ind, (samples, mask) in enumerate(zip(batch_samples.unbind(dim = 0), batch_mask.unbind(dim = 0))):
            if not torch.any(mask):
                continue

            sampled = self.sample_fn(rearrange(samples, '... -> 1 ...'), mask.sum().item())
            self.embed.data[ind][mask] = rearrange(sampled, '1 ... -> ...')

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code

        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, 'h ... d -> h (...) d')
        self.replace(batch_samples, batch_mask = expired_codes)

    @autocast(enabled = False)
    def forward(self, x):
        needs_codebook_dim = x.ndim < 4

        x = x.float()

        if needs_codebook_dim:
            x = rearrange(x, '... -> 1 ...')

        shape, dtype = x.shape, x.dtype
        flatten = rearrange(x, 'h ... d -> h (...) d')

        self.init_embed_(flatten)

        embed = self.embed if not self.learnable_codebook else self.embed.detach()

        dist = -torch.cdist(flatten, embed, p = 2)

        embed_ind = utils.gumbel_sample(dist, dim = -1, temperature = self.sample_codebook_temp)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = embed_ind.view(*shape[:-1])

        quantize = utils.batched_embedding(embed_ind, self.embed)

        if self.training:
            cluster_size = embed_onehot.sum(dim = 1)

            self.all_reduce_fn(cluster_size)
            utils.ema_inplace(self.cluster_size, cluster_size, self.decay)

            embed_sum = einsum('h n d, h n c -> h c d', flatten, embed_onehot)
            self.all_reduce_fn(embed_sum)

            cluster_size = utils.laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum()

            embed_normalized = self.embed_avg / rearrange(cluster_size, '... -> ... 1')
            self.embed.data.copy_(embed_normalized)
            self.expire_codes_(x)

        if needs_codebook_dim:
            quantize, embed_ind = map(lambda t: rearrange(t, '1 ... -> ...'), (quantize, embed_ind))

        return quantize, embed_ind

class CosineSimCodebook(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        kmeans_init = False,
        kmeans_iters = 10,
        decay = 0.8,
        eps = 1e-5,
        threshold_ema_dead_code = 2,
        learnable_codebook = False,
        sample_codebook_temp = 0.
    ):
        super().__init__()
        self.decay = decay

        if not kmeans_init:
            embed = F.normalize((utils.uniform_init(1, codebook_size, dim)), p = 2, dim = -1)
        else:
            embed = torch.zeros(1, codebook_size, dim)

        self.codebook_size = codebook_size

        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.sample_codebook_temp = sample_codebook_temp

        self.sample_fn = utils.batched_sample_vectors
        self.all_reduce_fn = utils.noop

        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.zeros(1, codebook_size))

        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer('embed', embed)

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.initted:
            return

        embed, cluster_size = utils.kmeans(
            data,
            self.codebook_size,
            self.kmeans_iters,
            use_cosine_sim = True,
            sample_fn = self.sample_fn,
            all_reduce_fn = self.all_reduce_fn
        )

        self.embed.data.copy_(embed)
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    def replace(self, batch_samples, batch_mask):
        batch_samples = F.normalize(batch_samples, p = 2, dim = -1)

        for ind, (samples, mask) in enumerate(zip(batch_samples.unbind(dim = 0), batch_mask.unbind(dim = 0))):
            if not torch.any(mask):
                continue

            sampled = self.sample_fn(rearrange(samples, '... -> 1 ...'), mask.sum().item())
            self.embed.data[ind][mask] = rearrange(sampled, '1 ... -> ...')

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code

        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, 'h ... d -> h (...) d')
        self.replace(batch_samples, batch_mask = expired_codes)

    @autocast(enabled = False)
    def forward(self, x):
        needs_codebook_dim = x.ndim < 4

        x = x.float()

        if needs_codebook_dim:
            x = rearrange(x, '... -> 1 ...')

        shape, dtype = x.shape, x.dtype

        flatten = rearrange(x, 'h ... d -> h (...) d')
        flatten = F.normalize(flatten, p = 2, dim = -1)

        self.init_embed_(flatten)

        embed = self.embed if not self.learnable_codebook else self.embed.detach()
        embed = F.normalize(embed, p = 2, dim = -1)

        dist = einsum('h n d, h c d -> h n c', flatten, embed)
        embed_ind = utils.gumbel_sample(dist, dim = -1, temperature = self.sample_codebook_temp)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = embed_ind.view(*shape[:-1])

        quantize = utils.batched_embedding(embed_ind, self.embed)

        if self.training:
            bins = embed_onehot.sum(dim = 1)
            self.all_reduce_fn(bins)

            utils.ema_inplace(self.cluster_size, bins, self.decay)

            zero_mask = (bins == 0)
            bins = bins.masked_fill(zero_mask, 1.)

            embed_sum = einsum('h n d, h n c -> h c d', flatten, embed_onehot)
            self.all_reduce_fn(embed_sum)

            embed_normalized = embed_sum / rearrange(bins, '... -> ... 1')
            embed_normalized = F.normalize(embed_normalized, p = 2, dim = -1)

            embed_normalized = torch.where(
                rearrange(zero_mask, '... -> ... 1'),
                embed,
                embed_normalized
            )

            utils.ema_inplace(self.embed, embed_normalized, self.decay)
            self.expire_codes_(x)

        if needs_codebook_dim:
            quantize, embed_ind = map(lambda t: rearrange(t, '1 ... -> ...'), (quantize, embed_ind))

        return quantize, embed_ind

# main class

class VectorQuantize(nn.Module):
    def __init__(
        self,
        codebook_size,
        codebook_dim = None,
        decay = 0.8,
        eps = 1e-5,
        kmeans_init = False,
        kmeans_iters = 10,
        use_cosine_sim = False,
        threshold_ema_dead_code = 0,
        channel_last = True,
        accept_image_fmap = False,
        commitment_weight = 1.,
        orthogonal_reg_weight = 0.,
        orthogonal_reg_active_codes_only = False,
        orthogonal_reg_max_codes = None,
        sample_codebook_temp = 0.,
        sync_codebook = True,
    ):
        super().__init__()

        self.eps = eps
        self.commitment_weight = commitment_weight

        has_codebook_orthogonal_loss = orthogonal_reg_weight > 0
        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes

        codebook_class = EuclideanCodebook if not use_cosine_sim else CosineSimCodebook

        self._codebook = codebook_class(
            dim = codebook_dim,
            codebook_size = codebook_size,
            kmeans_init = kmeans_init,
            kmeans_iters = kmeans_iters,
            decay = decay,
            eps = eps,
            threshold_ema_dead_code = threshold_ema_dead_code,
            learnable_codebook = has_codebook_orthogonal_loss,
            sample_codebook_temp = sample_codebook_temp,
            use_ddp = sync_codebook,
        )

        self.accept_image_fmap = accept_image_fmap
        self.channel_last = channel_last

    @property
    def codebook(self):
        codebook = self._codebook.embed
        if self.separate_codebook_per_head:
            return codebook

        return rearrange(codebook, '1 ... -> ...')
    
    def forward(self, x):
        shape, device = x.shape, x.device

        need_transpose = not self.channel_last and not self.accept_image_fmap

        if self.accept_image_fmap:
            height, width = x.shape[-2:]
            x = rearrange(x, 'b c h w -> b (h w) c')

        if need_transpose:
            x = rearrange(x, 'b d n -> b n d')

        quantize, embed_ind = self._codebook(x)

        if self.training:
            quantize = x + (quantize - x).detach()

        loss = torch.tensor([0.], device = device, requires_grad = self.training)

        if self.training:
            if self.commitment_weight > 0:
                commit_loss = F.mse_loss(quantize.detach(), x)
                loss = loss + commit_loss * self.commitment_weight

            if self.orthogonal_reg_weight > 0:
                codebook = self._codebook.embed

                if self.orthogonal_reg_active_codes_only:
                    # only calculate orthogonal loss for the activated codes for this batch
                    unique_code_ids = torch.unique(embed_ind)
                    codebook = codebook[unique_code_ids]

                num_codes = codebook.shape[0]
                if self.orthogonal_reg_max_codes is not None and num_codes > self.orthogonal_reg_max_codes:
                    rand_ids = torch.randperm(num_codes, device = device)[:self.orthogonal_reg_max_codes]
                    codebook = codebook[rand_ids]

                orthogonal_reg_loss = orthogonal_loss_fn(codebook)
                loss = loss + orthogonal_reg_loss * self.orthogonal_reg_weight

        if need_transpose:
            quantize = rearrange(quantize, 'b n d -> b d n')

        if self.accept_image_fmap:
            quantize = rearrange(quantize, 'b (h w) c -> b c h w', h = height, w = width)
            embed_ind = rearrange(embed_ind, 'b (h w) ... -> b h w ...', h = height, w = width)

        return quantize, loss, (None, None, embed_ind)
    
    def get_codebook_entry(self, indices, *kwargs):
        indices = indices.view(1, indices.size(0), -1)
        quantize = utils.batched_embedding(indices, self._codebook.embed).squeeze(0)
        return quantize


if __name__ == "__main__":
    x = torch.randn(10, 256, 32, 32)
    # x = torch.randn(10, 256, 120)

    quantizer = VectorQuantize(
        codebook_size = 1024,
        codebook_dim = 256,
        decay = 0.8,
        eps = 1e-5,
        kmeans_init = True,
        kmeans_iters = 10,
        use_cosine_sim = False,
        threshold_ema_dead_code = 0,
        channel_last = False,
        accept_image_fmap = True,
        commitment_weight = 1.,
        orthogonal_reg_weight = 0.,
        orthogonal_reg_active_codes_only = False,
        orthogonal_reg_max_codes = None,
        sample_codebook_temp = 0.,
        sync_codebook = False,
    )

    quantize, embed_ind, loss = quantizer(x)
    print(embed_ind, loss)

    indices = torch.randint(0, 1024, (10, 32, 32))
    quantize = quantizer.get_codebook_entry(indices, (10, 32, 32))
    print(quantize.size())