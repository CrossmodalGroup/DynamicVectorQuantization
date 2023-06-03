import math

import torch
import torch.nn as nn
from torch.nn import functional as F

class StackGPTConfig:
    """ base SatckGPT config, params common to all StackGPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, **kwargs):
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

class StackGPT(nn.Module):
    # using seperate embeddings for coarse-position and fine-position
    def __init__(self,
                 vocab_size, 
                 coarse_position_size,
                 fine_position_size,
                 segment_size = -1, # if segment_size <=0 means no segment embedding

                 block_size = None, 
                 position_layer = 12, 
                 content_layer = 12, 

                 n_head = 8, 
                 n_embd = 256, 

                 embd_pdrop = 0., 
                 resid_pdrop = 0., 
                 attn_pdrop = 0., 
    
                 content_pad_code = 1025,
                 coarse_position_pad_code = 257,
                 fine_position_pad_code = 1025,
                 
                 activate_pad_ignore = True, 
                 ):
        super().__init__()
        # configs
        n_unmasked = 0
        config = StackGPTConfig(
            vocab_size = vocab_size, coarse_position_size = coarse_position_size, fine_position_size = fine_position_size,
            block_size = block_size, embd_pdrop = embd_pdrop, resid_pdrop = resid_pdrop, attn_pdrop = attn_pdrop, 
            position_layer = position_layer, content_layer = content_layer, n_head = n_head, n_embd = n_embd, n_unmasked = n_unmasked
        )
        self.activate_segment = True if segment_size > 0 else False
        self.activate_pad_ignore = activate_pad_ignore
        self.coarse_position_pad_code = coarse_position_pad_code
        self.fine_position_pad_code = fine_position_pad_code
        self.content_pad_code = content_pad_code
        self.block_size = config.block_size
        self.config = config

        # input embedding stem
        ## position embeddings
        self.content_coarse_pos_emb = nn.Embedding(config.coarse_position_size, config.n_embd, padding_idx=coarse_position_pad_code)
        self.content_fine_pos_emb = nn.Embedding(config.fine_position_size, config.n_embd, padding_idx=fine_position_pad_code)
        ## content embeddings
        self.content_emb = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=content_pad_code)
        ## extra position embeddings to distinguish the order of elements in the sequence
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        ## extra segment embeddings to distinguish coarse- and grain- sequence
        if self.activate_segment:
            self.seg_emb = nn.Embedding(segment_size, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)

        # distinct transformer
        self.position_transformer = nn.Sequential(*[Block(config) for _ in range(config.position_layer)])
        self.content_transformer = nn.Sequential(*[Block(config) for _ in range(config.content_layer)])

        # prediction head 
        self.position_head = nn.Sequential(nn.LayerNorm(config.n_embd), nn.Linear(config.n_embd, config.fine_position_size, bias=False))
        self.content_head = nn.Sequential(nn.LayerNorm(config.n_embd), nn.Linear(config.n_embd, config.vocab_size, bias=False))
        
        # others
        self.apply(self._init_weights)
    
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
    
    def forward(self, coarse_content, fine_content, coarse_position, fine_position, coarse_seg, fine_seg, content_target=None, coarse_position_target=None, fine_position_target=None, **ignorekwargs):
        coarse_length = coarse_position.size(1)

        # content embeddings
        content = torch.cat([coarse_content, fine_content], dim=1)
        content_embeddings = self.content_emb(content[:, :-1])

        # position embeddings
        coarse_position_embeddings = self.content_coarse_pos_emb(coarse_position)
        fine_position_embeddings = self.content_fine_pos_emb(fine_position[:, :-1])
        position_embeddings = torch.cat([coarse_position_embeddings, fine_position_embeddings], dim=1)

        t = position_embeddings.shape[1]
        position_embeddings += self.pos_emb[:, :t, :]
        
        position_gpt_input = content_embeddings + position_embeddings

        # segment embeddings
        if self.activate_segment:
            segment = torch.cat([coarse_seg, fine_seg], dim=1)
            position_gpt_input += self.seg_emb(segment[:, :-1])

        # drop out for embedings 
        position_gpt_input = self.drop(position_gpt_input)

        # pass through position_transformer
        position_hidden = self.position_transformer(position_gpt_input)

        # pass through content_transformer
        update_coarse_position_embeddings = self.content_coarse_pos_emb(coarse_position[:, 1:])
        update_fine_position_embeddings = self.content_fine_pos_emb(fine_position)
        update_position_embeddings = torch.cat([update_coarse_position_embeddings, update_fine_position_embeddings], dim=1)
        content_gpt_input = position_hidden + update_position_embeddings
        content_hidden = self.content_transformer(content_gpt_input)
        
        # content head and position head
        content_logits = self.content_head(content_hidden)
        position_logits = self.position_head(position_hidden)

        if content_target is not None and coarse_position_target is not None and fine_position_target is not None:
            if self.activate_pad_ignore:
                coarse_position_logits = position_logits[:, :coarse_length-1]
                fine_position_logits = position_logits[:, coarse_length-1:]
                coarse_position_loss = F.cross_entropy(coarse_position_logits.contiguous().view(-1, coarse_position_logits.size(-1)), coarse_position_target.contiguous().view(-1), ignore_index=self.coarse_position_pad_code)
                fine_position_loss = F.cross_entropy(fine_position_logits.contiguous().view(-1, fine_position_logits.size(-1)), fine_position_target.contiguous().view(-1), ignore_index=self.fine_position_pad_code)
                position_loss = (coarse_position_loss + fine_position_loss) / 2
                content_loss = F.cross_entropy(content_logits.contiguous().view(-1, content_logits.size(-1)), content_target.contiguous().view(-1), ignore_index=self.content_pad_code)
            else:
                coarse_position_logits = position_logits[:, :coarse_length]
                fine_position_logits = position_logits[:, coarse_length:]
                coarse_position_loss = F.cross_entropy(coarse_position_logits.contiguous().view(-1, coarse_position_logits.size(-1)), coarse_position_target.contiguous().view(-1), ignore_index=self.coarse_position_pad_code)
                fine_position_loss = F.cross_entropy(fine_position_logits.contiguous().view(-1, fine_position_logits.size(-1)), fine_position_target.contiguous().view(-1), ignore_index=self.fine_position_pad_code)
                position_loss = (coarse_position_loss + fine_position_loss) / 2
                content_loss = F.cross_entropy(content_logits.contiguous().view(-1, content_logits.size(-1)), content_target.contiguous().view(-1))

            return {"position_loss": position_loss, "content_loss": content_loss, "coarse_position_loss": coarse_position_loss, "fine_position_loss": fine_position_loss}
        else:
            return {"position_logits": position_logits, "content_logits": content_logits}
    
    @torch.no_grad()
    def sample_coarse_position(self, coarse_content, coarse_position, coarse_seg):
        content_embeddings = self.content_emb(coarse_content)
        position_embeddings = self.content_coarse_pos_emb(coarse_position)

        position_gpt_input = content_embeddings + position_embeddings

        t = position_gpt_input.shape[1]
        position_gpt_input += self.pos_emb[:, :t, :]
        if self.activate_segment:
            position_gpt_input += self.seg_emb(coarse_seg)

        # pass through position_transformer
        position_hidden = self.position_transformer(position_gpt_input)
        position_logits = self.position_head(position_hidden)

        return position_hidden, position_logits
    
    @torch.no_grad()
    def sample_coarse_content(self, coarse_content=None, coarse_position=None, coarse_seg=None, position_hidden=None):
        if position_hidden is None:
            content_embeddings = self.content_emb(coarse_content)
            position_embeddings = self.content_coarse_pos_emb(coarse_position[:, :-1])

            position_gpt_input = content_embeddings + position_embeddings

            t = position_gpt_input.shape[1]
            position_gpt_input += self.pos_emb[:, :t, :]
            if self.activate_segment:
                position_gpt_input += self.seg_emb(coarse_seg[:, :-1])

            # pass through position_transformer
            position_hidden = self.position_transformer(position_gpt_input)
        
        # pass through content_transformer
        update_position_embeddings = self.content_coarse_pos_emb(coarse_position[:, 1:])
        content_gpt_input = position_hidden + update_position_embeddings
        content_hidden = self.content_transformer(content_gpt_input)
        
        # content head and position head
        content_logits = self.content_head(content_hidden)

        return content_hidden, content_logits

    def sample_fine_position(self, coarse_content, fine_content, coarse_position, fine_position, coarse_seg, fine_seg):
        content = torch.cat([coarse_content, fine_content], dim=1)
        content_embeddings = self.content_emb(content)

        coarse_position_embeddings = self.content_coarse_pos_emb(coarse_position)
        fine_position_embeddings = self.content_fine_pos_emb(fine_position)
        position_embeddings = torch.cat([coarse_position_embeddings, fine_position_embeddings], dim=1)

        t = position_embeddings.shape[1]
        position_embeddings += self.pos_emb[:, :t, :]
        position_gpt_input = content_embeddings + position_embeddings

        if self.activate_segment:
            if fine_seg is not None:
                segment = torch.cat([coarse_seg, fine_seg], dim=1)
            else:
                segment = coarse_seg
            position_gpt_input += self.seg_emb(segment)
        position_gpt_input = self.drop(position_gpt_input)

        # pass through position_transformer
        position_hidden = self.position_transformer(position_gpt_input)

        position_logits = self.position_head(position_hidden)

        return position_hidden, position_logits
    
    def sample_fine_content(self, coarse_content, fine_content, coarse_position, fine_position, coarse_seg, fine_seg, position_hidden=None):
        if position_hidden is None:
            content = torch.cat([coarse_content, fine_content], dim=1)
            content_embeddings = self.content_emb(content)

            coarse_position_embeddings = self.content_coarse_pos_emb(coarse_position)
            fine_position_embeddings = self.content_fine_pos_emb(fine_position[:, :-1])
            position_embeddings = torch.cat([coarse_position_embeddings, fine_position_embeddings], dim=1)

            t = position_embeddings.shape[1]
            position_embeddings += self.pos_emb[:, :t, :]
            position_gpt_input = content_embeddings + position_embeddings

            if self.activate_segment:
                if fine_seg is not None:
                    segment = torch.cat([coarse_seg, fine_seg], dim=1)
                else:
                    segment = coarse_seg
                position_gpt_input += self.seg_emb(segment)
            position_gpt_input = self.drop(position_gpt_input)

            # pass through position_transformer
            position_hidden = self.position_transformer(position_gpt_input)
        
        # pass through content_transformer
        update_coarse_position_embeddings = self.content_coarse_pos_emb(coarse_position)
        update_fine_position_embeddings = self.content_fine_pos_emb(fine_position[:, 1:])
        update_position_embeddings = torch.cat([update_coarse_position_embeddings, update_fine_position_embeddings], dim=1)

        content_gpt_input = position_hidden + update_position_embeddings
        content_hidden = self.content_transformer(content_gpt_input)
        
        # content head and position head
        content_logits = self.content_head(content_hidden)

        return content_hidden, content_logits