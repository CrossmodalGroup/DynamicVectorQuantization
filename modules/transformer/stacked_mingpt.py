# using two seperate transformer to model position and value distribution respectively

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class PositionAwareGPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, position_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.position_size = position_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        mask = torch.tril(torch.ones(config.block_size,
                                     config.block_size))
        if hasattr(config, "n_unmasked"):
            mask[:config.n_unmasked, :config.n_unmasked] = 1
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        present = torch.stack((k, v))
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if layer_past is None:
            att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, present   # TODO: check that this does not break anything

class Block(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),  # nice
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, layer_past=None, return_present=False):
        # TODO: check that training still works
        if return_present: assert not self.training
        # layer past: tuple of length two with B, nh, T, hs
        attn, present = self.attn(self.ln1(x), layer_past=layer_past)

        x = x + attn
        x = x + self.mlp(self.ln2(x))
        if layer_past is not None or return_present:
            return x, present
        return x

# firstly sample position, secondly sample value
class StackedPositionGPT(nn.Module):
    def __init__(self, vocab_size, position_size, block_size, position_layer=12, 
                 value_layer=12, n_head=8, n_embd=256, embd_pdrop=0., resid_pdrop=0., 
                 attn_pdrop=0., n_unmasked=0, add_absolute_position=True):
        super().__init__()
        # configs
        config = PositionAwareGPTConfig(
                           vocab_size=vocab_size, block_size=block_size, position_size=position_size,
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                           position_layer=position_layer, value_layer=value_layer, n_head=n_head, 
                           n_embd=n_embd, n_unmasked=n_unmasked)

        # input embedding stem
        ## position embeddings
        self.value_pos_emb = nn.Embedding(config.position_size, config.n_embd)
        ## value embeddings
        self.value_emb = nn.Embedding(config.vocab_size, config.n_embd)
        ## extra position embeddings for distinguish different input element
        self.add_absolute_position = add_absolute_position
        if self.add_absolute_position:
            self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        # distinct transformer
        self.position_transformer = nn.Sequential(*[Block(config) for _ in range(config.position_layer)])
        self.value_transformer = nn.Sequential(*[Block(config) for _ in range(config.value_layer)])

        # prediction head 
        self.position_head = nn.Sequential(
            nn.LayerNorm(config.n_embd),
            nn.Linear(config.n_embd, config.position_size, bias=False)
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(config.n_embd),
            nn.Linear(config.n_embd, config.vocab_size, bias=False)
        )
        
        # others
        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))
    
    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, idx, pos_idx, idx_target=None, pos_idx_target=None, **ignorekwargs):
        r"""
        idx: [<value_sos>, <v1>, <v2>, <v3>, <v4>]
        pos_idx: [<pos_sos>, <p1>, <p2>, <p3>, <p4>]
        idx_target: [<v1>, <v2>, <v3>, <v4>]
        pos_idx_target: [<p1>, <p2>, <p3>, <p4>]
        """
        # first pass through position_transformer
        ## embed both position and value, drop the last element
        value_embeddings = self.value_emb(idx[:, :-1])  # [<value_sos>, <v1>, <v2>, <v3>]
        position_embeddings = self.value_pos_emb(pos_idx[:, :-1])  # [<pos_sos>, <p1>, <p2>, <p3>]
        token_embeddings = value_embeddings + position_embeddings

        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        if self.add_absolute_position:
            abs_position_embeddings = self.pos_emb[:, :t, :]
            position_gpt_input = self.drop(token_embeddings + abs_position_embeddings)
        else:
            position_gpt_input = self.drop(token_embeddings)
        
        position_hidden = self.position_transformer(position_gpt_input) # for predict [<p1>, <p2>, <p3>, <p4>]

        # pass through value_transformer
        ## [<p1>, <p2>, <p3>, <p4>] for predicting values
        update_position_embeddings = self.value_pos_emb(pos_idx[:, 1:])
        value_gpt_input = position_hidden + update_position_embeddings

        value_hidden = self.value_transformer(value_gpt_input) # for predict [<v1>, <v2>, <v3>, <v4>]

        # position head and value head
        position_logits = self.position_head(position_hidden)
        value_logits = self.value_head(value_hidden)

        if idx_target is not None and pos_idx_target is not None:
            position_loss = F.cross_entropy(
                position_logits.contiguous().view(-1, position_logits.size(-1)), pos_idx_target.contiguous().view(-1)
            )
            value_loss = F.cross_entropy(
                value_logits.contiguous().view(-1, value_logits.size(-1)), idx_target.contiguous().view(-1)
            )

            return {
                "position_loss": position_loss,
                "value_loss": value_loss
            }
        else:
            return {
                "position_logits": position_logits,
                "value_logits": value_logits
            }

    @torch.no_grad()
    def sample_position(self, idx, pos_idx):
        # first pass through position_transformer
        ## embed both position and value, drop the last element
        value_embeddings = self.value_emb(idx)  # [<value_sos>]
        position_embeddings = self.value_pos_emb(pos_idx)  # [<pos_sos>]
        token_embeddings = value_embeddings + position_embeddings

        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        if self.add_absolute_position:
            abs_position_embeddings = self.pos_emb[:, :t, :]
            position_gpt_input = self.drop(token_embeddings + abs_position_embeddings)
        else:
            position_gpt_input = self.drop(token_embeddings)
        
        position_hidden = self.position_transformer(position_gpt_input) # for predict [<p1>]
        position_logits = self.position_head(position_hidden)

        return position_logits
        
    @torch.no_grad()
    def sample_value(self, idx, pos_idx):
        # first pass through position_transformer
        ## embed both position and value, drop the last element
        value_embeddings = self.value_emb(idx)  # [<value_sos>]
        position_embeddings = self.value_pos_emb(pos_idx[:, :-1])  # [<pos_sos>, <p1>]
        token_embeddings = value_embeddings + position_embeddings

        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        if self.add_absolute_position:
            abs_position_embeddings = self.pos_emb[:, :t, :]
            position_gpt_input = self.drop(token_embeddings + abs_position_embeddings)
        else:
            position_gpt_input = self.drop(token_embeddings)
        
        position_hidden = self.position_transformer(position_gpt_input) # for predict [<p1>]

        # pass through value_transformer
        ## [<p1>] for predicting values
        update_position_embeddings = self.value_pos_emb(pos_idx[:, 1:])
        value_gpt_input = position_hidden + update_position_embeddings

        value_hidden = self.value_transformer(value_gpt_input) # for predict [<v1>]
        value_logits = self.value_head(value_hidden)

        return value_logits

# firstly sample value, secondly sample position 
class ReverseStackedPositionGPT(nn.Module):
    def __init__(self, vocab_size, position_size, block_size, position_layer=12, 
                 value_layer=12, n_head=8, n_embd=256, embd_pdrop=0., resid_pdrop=0., 
                 attn_pdrop=0., n_unmasked=0, add_absolute_position=True):
        super().__init__()
        # configs
        config = PositionAwareGPTConfig(
                           vocab_size=vocab_size, block_size=block_size, position_size=position_size,
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                           position_layer=position_layer, value_layer=value_layer, n_head=n_head, 
                           n_embd=n_embd, n_unmasked=n_unmasked)

        # input embedding stem
        ## position embeddings
        self.value_pos_emb = nn.Embedding(config.position_size, config.n_embd)
        ## value embeddings
        self.value_emb = nn.Embedding(config.vocab_size, config.n_embd)
        ## extra position embeddings for distinguish different input element
        self.add_absolute_position = add_absolute_position
        if self.add_absolute_position:
            self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        # distinct transformer
        self.position_transformer = nn.Sequential(*[Block(config) for _ in range(config.position_layer)])
        self.value_transformer = nn.Sequential(*[Block(config) for _ in range(config.value_layer)])

        # prediction head 
        self.position_head = nn.Sequential(
            nn.LayerNorm(config.n_embd),
            nn.Linear(config.n_embd, config.position_size, bias=False)
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(config.n_embd),
            nn.Linear(config.n_embd, config.vocab_size, bias=False)
        )
        
        # others
        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))
    
    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, idx, pos_idx, idx_target=None, pos_idx_target=None, **ignorekwargs):
        r"""
        idx: [<value_sos>, <v1>, <v2>, <v3>, <v4>]
        pos_idx: [<pos_sos>, <p1>, <p2>, <p3>, <p4>]
        idx_target: [<v1>, <v2>, <v3>, <v4>]
        pos_idx_target: [<p1>, <p2>, <p3>, <p4>]
        """
        # first pass through value_transformer
        ## embed both position and value, drop the last element
        value_embeddings = self.value_emb(idx[:, :-1])  # [<value_sos>, <v1>, <v2>, <v3>]
        position_embeddings = self.value_pos_emb(pos_idx[:, :-1])  # [<pos_sos>, <p1>, <p2>, <p3>]
        token_embeddings = value_embeddings + position_embeddings

        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        if self.add_absolute_position:
            abs_position_embeddings = self.pos_emb[:, :t, :]
            value_gpt_input = self.drop(token_embeddings + abs_position_embeddings)
        else:
            value_gpt_input = self.drop(token_embeddings)
            
        value_hidden = self.value_transformer(value_gpt_input)
        
        # pass through position_gpt
        ## [<v1>,<v2>,<v3>,<v4>] for predicting positions
        update_value_embeddings = self.value_emb(idx[:, 1:])
        position_gpt_input = value_hidden + update_value_embeddings
        
        position_hidden = self.position_transformer(position_gpt_input)  # for predict [<p1>,<p2>,<p3>,<p4>]
        
        # value head and position head
        value_logits = self.value_head(value_hidden)
        position_logits = self.position_head(position_hidden)
        
        if idx_target is not None and pos_idx_target is not None:
            position_loss = F.cross_entropy(
                position_logits.contiguous().view(-1, position_logits.size(-1)), pos_idx_target.contiguous().view(-1)
            )
            value_loss = F.cross_entropy(
                value_logits.contiguous().view(-1, value_logits.size(-1)), idx_target.contiguous().view(-1)
            )

            return {
                "position_loss": position_loss,
                "value_loss": value_loss
            }
        else:
            return {
                "position_logits": position_logits,
                "value_logits": value_logits
            }
    
    @torch.no_grad()
    def sample_value(self, idx, pos_idx):
        # first pass through value_transformer
        ## embed both position and value, drop the last element
        value_embeddings = self.value_emb(idx)  # [<value_sos>]
        position_embeddings = self.value_pos_emb(pos_idx)  # [<pos_sos>]
        token_embeddings = value_embeddings + position_embeddings

        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        if self.add_absolute_position:
            abs_position_embeddings = self.pos_emb[:, :t, :]
            value_gpt_input = self.drop(token_embeddings + abs_position_embeddings)
        else:
            value_gpt_input = self.drop(token_embeddings)
        
        value_hidden = self.value_transformer(value_gpt_input)
        value_logits = self.value_head(value_hidden)
        
        return value_logits
    
    @torch.no_grad()
    def sample_position(self, idx, pos_idx):
        # first pass through value_transformer
        ## embed both position and value, drop the last element
        value_embeddings = self.value_emb(idx[:, :-1])  # [<value_sos>, <v1>]
        position_embeddings = self.value_pos_emb(pos_idx)  # [<pos_sos>]
        token_embeddings = value_embeddings + position_embeddings

        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        if self.add_absolute_position:
            abs_position_embeddings = self.pos_emb[:, :t, :]
            value_gpt_input = self.drop(token_embeddings + abs_position_embeddings)
        else:
            value_gpt_input = self.drop(token_embeddings)
        
        value_hidden = self.value_transformer(value_gpt_input)
        
        # pass through position_gpt
        ## [<v1>,<v2>,<v3>,<v4>] for predicting positions
        update_value_embeddings = self.value_emb(idx[:, 1:])
        position_gpt_input = value_hidden + update_value_embeddings
        
        position_hidden = self.position_transformer(position_gpt_input)  # for predict [<p1>,<p2>,<p3>,<p4>]
        position_logits = self.position_head(position_hidden)
        
        return position_logits
    

if __name__ == "__main__":
    model = ReverseStackedPositionGPT(
        vocab_size=1024, position_size=256, block_size=257, 
        position_layer=3, value_layer=3, n_head=4, n_embd=256,
        embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., n_unmasked=0,
        add_absolute_position=True
    )
    
    idx = torch.randint(0, 1024, (1, 257))
    pos_idx = torch.randint(0, 256, (1, 257))
    
    output = model(idx, pos_idx, idx[:, 1:], pos_idx[:, 1:])

    print(output["position_loss"], output["value_loss"])