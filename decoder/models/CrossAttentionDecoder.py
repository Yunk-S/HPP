import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, prompt_feat, return_context=False, gates=None):
        B, N, C = x.shape
        B, M, C = prompt_feat.shape  # M=1 for prompt

        q = self.q_proj(prompt_feat).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, 1, N)
        if gates is not None:
            if gates.shape != (B, N):
                raise ValueError('gates must have shape [B,N]')
            attn = attn.float() + gates.float().clamp_min(1e-12).log()[:, None, None, :]
        attn = attn.softmax(dim=-1).to(v.dtype)
        attn = self.attn_drop(attn)

        context = (attn @ v).transpose(1, 2).reshape(B, M, C)  # (B, 1, C)
        context = self.proj(context)
        context = self.proj_drop(context)

        if return_context:
            return context

        x = context.expand(-1, N, -1)  # (B, N, C)
        return x


class CrossAttentionPointsQ(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, prompt_feat, gates=None):
        B, N, C = x.shape
        B, M, C = prompt_feat.shape  # M=1

        q = self.q_proj(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)      # (B,h,N,d)
        k = self.k_proj(prompt_feat).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # (B,h,M,d)
        v = self.v_proj(prompt_feat).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # (B,h,M,d)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B,h,N,M)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B,N,C)
        out = self.proj(out)
        out = self.proj_drop(out)
        if gates is not None:
            out = out * gates.to(out.dtype).unsqueeze(-1)
        return out


class CrossAttentionDecoder(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True,
                 attn_drop=0., proj_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CrossAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop
        )
        self.attn_points_q = CrossAttentionPointsQ(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(proj_drop)
        )
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, prompt_feat, gates=None, return_prompt=False):
        residual_prompt = prompt_feat
        x_norm = self.norm1(x)
        context = self.attn(x_norm, prompt_feat, return_context=True, gates=gates)  # (B,1,C)
        prompt_feat = residual_prompt + self.drop_path(context)

        residual_x = x
        out_points = self.attn_points_q(x_norm, prompt_feat, gates=gates)  # (B,N,C)
        x = residual_x + self.drop_path(out_points)

        # MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + self.drop_path(x)
        return (x, prompt_feat) if return_prompt else x


class Decoder(nn.Module):
    def __init__(self, dim, num_heads, num_layers=4, mlp_ratio=4., qkv_bias=True,
                 attn_drop=0., proj_drop=0., drop_path=0., gate_aware=False):
        super().__init__()
        self.gate_aware = gate_aware
        self.decoder_blocks = nn.ModuleList([
            CrossAttentionDecoder(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                drop_path=drop_path
            ) for _ in range(num_layers)
        ])

    def forward(self, x, prompt_feat, gates=None):
        if gates is not None and not self.gate_aware:
            raise ValueError('gates require gate_aware=True')
        for block in self.decoder_blocks:
            if self.gate_aware:
                x, prompt_feat = block(x, prompt_feat, gates=gates, return_prompt=True)
            else:
                x = block(x, prompt_feat)
        return x


def make(cfg):
    return Decoder(
        dim=cfg.decoder.hidden_dim,
        num_heads=cfg.decoder.num_heads,
        num_layers=cfg.decoder.num_layers,
        mlp_ratio=cfg.decoder.mlp_ratio,
        qkv_bias=cfg.decoder.qkv_bias,
        attn_drop=cfg.decoder.attn_drop,
        proj_drop=cfg.decoder.proj_drop,
        drop_path=cfg.decoder.drop_path,
        gate_aware=cfg.get("model_variant", "legacy") == "hyperseg-h"
    ) 

# Public descriptive aliases; parameter names remain checkpoint compatible.
DecoderBlockGateAware = CrossAttentionDecoder
CrossAttentionPointsQGateAware = CrossAttentionPointsQ
