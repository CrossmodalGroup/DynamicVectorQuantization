import torch
from torch import nn, einsum
import torch.nn.functional as F
import torch.distributed as distributed
from einops import rearrange, repeat

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def noop(*args, **kwargs):
    pass

def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def uniform_init(*shape):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    if temperature == 0:
        return t.argmax(dim = dim)

    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha = (1 - decay))

def laplace_smoothing(x, n_categories, eps = 1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)

def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device = device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device = device)

    return samples[indices]

def batched_sample_vectors(samples, num):
    return torch.stack([sample_vectors(sample, num) for sample in samples.unbind(dim = 0)], dim = 0)

def pad_shape(shape, size, dim = 0):
    return [size if i == dim else s for i, s in enumerate(shape)]

def sample_multinomial(total_count, probs):
    device = probs.device
    probs = probs.cpu()

    total_count = probs.new_full((), total_count)
    remainder = probs.new_ones(())
    sample = torch.empty_like(probs, dtype = torch.long)

    for i, p in enumerate(probs):
        s = torch.binomial(total_count, p / remainder)
        sample[i] = s
        total_count -= s
        remainder -= p

    return sample.to(device)

def all_gather_sizes(x, dim):
    size = torch.tensor(x.shape[dim], dtype = torch.long, device = x.device)
    all_sizes = [torch.empty_like(size) for _ in range(distributed.get_world_size())]
    distributed.all_gather(all_sizes, size)

    return torch.stack(all_sizes)

def all_gather_variably_sized(x, sizes, dim = 0):
    rank = distributed.get_rank()
    all_x = []

    for i, size in enumerate(sizes):
        t = x if i == rank else x.new_empty(pad_shape(x.shape, size, dim))
        distributed.broadcast(t, src = i, async_op = True)
        all_x.append(t)

    distributed.barrier()
    return all_x

def sample_vectors_distributed(local_samples, num):
    rank = distributed.get_rank()
    all_num_samples = all_gather_sizes(local_samples, dim = 0)

    if rank == 0:
        samples_per_rank = sample_multinomial(num, all_num_samples / all_num_samples.sum())
    else:
        samples_per_rank = torch.empty_like(all_num_samples)

    distributed.broadcast(samples_per_rank, src = 0)
    samples_per_rank = samples_per_rank.tolist()

    local_samples = batched_sample_vectors(local_samples, samples_per_rank[rank])
    all_samples = all_gather_variably_sized(local_samples, samples_per_rank, dim = 0)
    return torch.cat(all_samples, dim = 0)

def batched_bincount(x, *, minlength):
    batch, dtype, device = x.shape[0], x.dtype, x.device
    target = torch.zeros(batch, minlength, dtype = dtype, device = device)
    values = torch.ones_like(x)
    target.scatter_add_(-1, x, values)
    return target

def kmeans(
    samples,
    num_clusters,
    num_iters = 10,
    use_cosine_sim = False,
    sample_fn = batched_sample_vectors,
    all_reduce_fn = noop
):
    num_codebooks, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device

    means = sample_fn(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ rearrange(means, 'h n d -> h d n')
        else:
            dists = -torch.cdist(samples, means, p = 2)

        buckets = torch.argmax(dists, dim = -1)
        bins = batched_bincount(buckets, minlength = num_clusters)
        all_reduce_fn(bins)

        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_codebooks, num_clusters, dim, dtype = dtype)

        new_means.scatter_add_(1, repeat(buckets, 'h n -> h n d', d = dim), samples)
        new_means = new_means / rearrange(bins_min_clamped, '... -> ... 1')
        all_reduce_fn(new_means)

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(
            rearrange(zero_mask, '... -> ... 1'),
            means,
            new_means
        )

    return means, bins

def batched_embedding(indices, embeds):
    batch, dim = indices.shape[1], embeds.shape[-1]
    indices = repeat(indices, 'h b n -> h b n d', d = dim)
    embeds = repeat(embeds, 'h c d -> h b c d', b = batch)
    return embeds.gather(2, indices)
