import torch
import torch.nn as nn

# 我们希望自注意力中的信息流向 bias 为 unmasked token -> masked token
class MaskSelfAttention_SquareGrowth(nn.Module):
    def __init__(self, dim, num_heads, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, h, mask=None):
        # mask (_type_, optional): [batch_size, length]
        B, N, C = h.shape
        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)).contiguous() * self.scale       
        attn = attn.softmax(dim=-1)
        
        if mask is not None:
            unsqueezed_mask = mask.unsqueeze(-2).unsqueeze(-2)
            attn = attn * unsqueezed_mask
            
            # update mask with SquareGrowth
            new_mask = torch.sqrt(mask)

        attn = self.attn_drop(attn)
        h = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        h = self.proj(h)
        h = self.proj_drop(h)
        return h, new_mask
    

if __name__ == "__main__":
    batch_size = 1
    dim = 256
    height = 2
    model = MaskSelfAttention_SquareGrowth(dim=256, num_heads=4)
    
    h = torch.randn(batch_size, height*height, 256)  # (10,256,16,16)
    mask = torch.randint(0, 2, (batch_size, height*height))
    print(mask)
    
    new_mask = mask + 0.02 * (1 - mask)
    print(new_mask)

    
    model(h, new_mask)