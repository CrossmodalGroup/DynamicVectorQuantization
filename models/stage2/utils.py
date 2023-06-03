import math
import torch
from functools import partial

# step scheduler
def fn(warmup_steps, max_steps, multipler_min, step):
    if step < warmup_steps:  # linear warmup
        return float(step) / float(max(1, warmup_steps))
    else:  # cosine learning rate schedule
        multipler = 0.5 * (math.cos((step - warmup_steps) / (max_steps - warmup_steps) * math.pi) + 1)
        return max(multipler, multipler_min)

def learning_rate_schedule(warmup_steps, max_steps, multipler_min):
    return partial(fn, warmup_steps, max_steps, multipler_min)

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

# commonly used sample functions
def top_k_logits(logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

def top_p_logits(probs, p):    
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    sorted_idx_remove_cond = cum_probs >= p
    
    sorted_idx_remove_cond[..., 1:] = sorted_idx_remove_cond[..., :-1].clone()
    sorted_idx_remove_cond[..., 0] = 0
    
    indices_to_remove = sorted_idx_remove_cond.scatter(-1, sorted_indices, sorted_idx_remove_cond)
    probs = probs.masked_fill(indices_to_remove, 0.0)
    norm_probs = probs / torch.sum(probs, dim=-1, keepdim=True)
    return norm_probs