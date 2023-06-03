import torch
import torch.nn as nn
import math
import numpy as np
from einops import rearrange

# input is (B,C,H,W), output is (B,HW,C)
def build_position_embed(embed_type='learned', feats_dim=512, dropout=0., n_row=16):
    if embed_type == 'sine-1d':
        pos_embed = PositionalEncoding1d(emb_dim=feats_dim, dropout=dropout)
    elif embed_type == "sine-2d":
        pos_embed = PositionalEncoding2d(emb_dim=feats_dim, dropout=dropout)
    elif embed_type == "learned-2d":
        pos_embed = PositionEmbeddingLearned(n_row=n_row, feats_dim=feats_dim, dropout=dropout)
    else:
        raise ValueError(f"nor supported {embed_type}")
    return pos_embed


######################################################################################
# 1D position embedding
######################################################################################
class PositionalEncoding1d(nn.Module):
    def __init__(self, emb_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding1d, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = rearrange(x, "B C H W -> B (H W) C").contiguous()
        x = x + self.pe[:x.size(0), :].to(x.device)
        return self.dropout(x)

######################################################################################
# 2D position embedding
######################################################################################
class PositionalEncoding2d(nn.Module):
    def __init__(self, emb_dim, dropout, max_len=5000):
        super(PositionalEncoding2d, self).__init__()
        
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, (emb_dim//2), 2).float() * (-math.log(10000.0) / (emb_dim//2)))
        pe_x = torch.zeros(max_len, emb_dim//2)
        pe_x[:, 0::2] = torch.sin(position * div_term)
        pe_x[:, 1::2] = torch.cos(position * div_term)
        
        pe_y = torch.zeros(max_len, emb_dim//2)
        pe_y[:, 1::2] = torch.sin(position * div_term)
        pe_y[:, 0::2] = torch.cos(position * div_term)

        self.register_buffer('pe_x', pe_x)
        self.register_buffer('pe_y', pe_y)

    def forward(self, x):
        _, _, h, w = x.shape
        add_x = self.pe_x[:h, :].unsqueeze(1).repeat(1,w,1)
        add_y = self.pe_y[:w, :].unsqueeze(0).repeat(h,1,1)
        add = torch.cat([add_x, add_y], dim=-1) #shape: h x w x dim 
        add = add.permute(2, 0, 1).unsqueeze(0)

        x = x + add
        x = rearrange(x, "B C H W -> B (H W) C").contiguous()
        return self.dropout(x)

class PositionEmbeddingLearned(nn.Module):
    """
    This is a learned version of the position embedding
    """
    def __init__(self, n_row, feats_dim, dropout, n_col=None):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        n_col = n_col if n_col is not None else n_row
        self.row_embed = nn.Embedding(n_row, feats_dim)
        self.col_embed = nn.Embedding(n_col, feats_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i).unsqueeze(0).repeat(h, 1, 1)
        y_emb = self.row_embed(j).unsqueeze(1).repeat(1, w, 1)
        pos = (x_emb + y_emb).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)

        x = x + pos
        x = rearrange(x, "B C H W -> B (H W) C").contiguous()
        return self.dropout(x)