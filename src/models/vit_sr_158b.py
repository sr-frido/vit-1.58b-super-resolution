import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.quant.bitlinear import BitLinear


class PatchEmbed(nn.Module):
    """
    Conv stem: LR image -> token map (B, N, C)
    """
    def __init__(self, in_chans=3, embed_dim=192, patch_size=1):
        super().__init__()
        # patch_size=1 keeps full LR resolution as tokens
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)

    def forward(self, x):
        # x: (B,3,H,W) -> (B, C, H, W) -> (B, N, C)
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        return x, (H, W)


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, act_bits=8, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = BitLinear(dim, hidden, bias=True, act_bits=act_bits)
        self.act = nn.GELU()
        self.fc2 = BitLinear(hidden, dim, bias=True, act_bits=act_bits)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=6, act_bits=8, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = BitLinear(dim, dim * 3, bias=True, act_bits=act_bits)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = BitLinear(dim, dim, bias=True, act_bits=act_bits)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)  # (B,N,3C)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B,h,N,d)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q.float() @ k.float().transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1).to(q.dtype)
        attn = self.attn_drop(attn)

        x = attn @ v  # (B,h,N,d)
        x = x.transpose(1, 2).contiguous().view(B, N, C)  # (B,N,C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, act_bits=8, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, act_bits=act_bits, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, act_bits=act_bits, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))  # pre-norm
        x = x + self.mlp(self.norm2(x))
        return x


class UpsampleHead(nn.Module):
    """
    Tokens (B,N,C) -> feature map (B,C,H,W) -> PixelShuffle -> SR image (B,3, sH, sW)
    """
    def __init__(self, embed_dim=192, scale=4, out_chans=3):
        super().__init__()
        self.scale = scale
        # Prepare channels for pixelshuffle: C -> out_chans*(scale^2)
        self.conv = nn.Conv2d(embed_dim, out_chans * (scale * scale), kernel_size=3, padding=1)
        self.ps = nn.PixelShuffle(scale)

    def forward(self, x, hw):
        H, W = hw
        B, N, C = x.shape
        feat = x.transpose(1, 2).contiguous().view(B, C, H, W)  # (B,C,H,W)
        out = self.conv(feat)
        out = self.ps(out)
        return out


class ViTSR158b(nn.Module):
    """
    Minimal ViT-based SR model:
    LR -> tokens -> transformer blocks -> pixelshuffle head
    """
    def __init__(
        self,
        img_channels=3,
        embed_dim=192,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        scale=4,
        act_bits=8,
        drop=0.0,
        attn_drop=0.0,
    ):
        super().__init__()
        self.scale = scale

        self.patch_embed = PatchEmbed(in_chans=img_channels, embed_dim=embed_dim, patch_size=1)

        # 2D learned positional embedding (simple): we do it as a maximum grid and interpolate
        self.pos_embed = None
        self.pos_drop = nn.Dropout(drop)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio=mlp_ratio, act_bits=act_bits, drop=drop, attn_drop=attn_drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.head = UpsampleHead(embed_dim=embed_dim, scale=scale, out_chans=3)

    def _get_2d_sincos_pos_embed(self, dim, grid_h, grid_w, device):
        # fixed sincos positional embedding (no params) to avoid max-size issues
        # returns (1, N, dim)
        assert dim % 4 == 0
        y = torch.arange(grid_h, device=device, dtype=torch.float32)
        x = torch.arange(grid_w, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing="ij")  # (H,W)
        omega = torch.arange(dim // 4, device=device, dtype=torch.float32) / (dim // 4)
        omega = 1.0 / (10000 ** omega)  # (dim/4,)

        out_y = yy.reshape(-1, 1) * omega.reshape(1, -1)
        out_x = xx.reshape(-1, 1) * omega.reshape(1, -1)

        pos = torch.cat([torch.sin(out_x), torch.cos(out_x), torch.sin(out_y), torch.cos(out_y)], dim=1)
        return pos.unsqueeze(0)  # (1,N,dim)

    def forward(self, lr):
        # lr: (B,3,h,w) in [0,1]
        x, (H, W) = self.patch_embed(lr)  # (B,N,C)

        pos = self._get_2d_sincos_pos_embed(x.shape[-1], H, W, x.device)
        x = x + pos
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # Residual prediction (HR)
        res = self.head(x, (H, W))  # (B,3,H*s,W*s)

        # Bicubic upsample baseline
        base = F.interpolate(lr, scale_factor=self.scale, mode="bicubic", align_corners=False)

        # Final SR
        sr = base + res
        return sr