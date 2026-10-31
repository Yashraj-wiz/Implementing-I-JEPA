from __future__ import annotations

import copy
import math

import torch
from torch import nn
import torch.nn.functional as F

from .masks import MaskBatch


def gather_tokens(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather [B, K] patch indices from [B, N, D] tokens."""
    if tokens.ndim != 3 or indices.ndim != 2 or tokens.size(0) != indices.size(0):
        raise ValueError("expected tokens [B,N,D] and indices [B,K]")
    return torch.gather(tokens, 1, indices.unsqueeze(-1).expand(-1, -1, tokens.size(-1)))


def sincos_2d(grid_size: int, dim: int) -> torch.Tensor:
    if dim % 4:
        raise ValueError("embedding dimension must be divisible by 4")
    y, x = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing="ij")
    omega = torch.arange(dim // 4, dtype=torch.float32) / (dim // 4)
    omega = 1.0 / (10000**omega)
    x = x.flatten().float()[:, None] * omega[None, :]
    y = y.flatten().float()[:, None] * omega[None, :]
    return torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=1).unsqueeze(0)


class MLP(nn.Module):
    def __init__(self, dim: int, ratio: float = 4.0) -> None:
        super().__init__()
        hidden = int(dim * ratio)
        self.layers = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.norm1(x)
        x = x + self.attn(q, q, q, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.grid_size = image_size // patch_size
        self.dim = dim
        self.patch_embed = nn.Conv2d(3, dim, patch_size, patch_size)
        self.register_buffer("pos_embed", sincos_2d(self.grid_size, dim), persistent=True)
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor, indices: torch.Tensor | None = None) -> torch.Tensor:
        x = self.patch_embed(images).flatten(2).transpose(1, 2)
        x = x + self.pos_embed.to(dtype=x.dtype)
        if indices is not None:
            x = gather_tokens(x, indices)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class Predictor(nn.Module):
    def __init__(
        self,
        grid_size: int,
        encoder_dim: int,
        predictor_dim: int,
        depth: int,
        heads: int,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.context_proj = nn.Linear(encoder_dim, predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.register_buffer("pos_embed", sincos_2d(grid_size, predictor_dim), persistent=True)
        self.blocks = nn.ModuleList([Block(predictor_dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(predictor_dim)
        self.output_proj = nn.Linear(predictor_dim, encoder_dim)

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        # target_indices: [M, B, Nt]; output: [M, B, Nt, Denc]
        num_targets, batch_size, target_length = target_indices.shape
        pos = self.pos_embed.expand(batch_size, -1, -1).to(dtype=context_tokens.dtype)
        context_pos = gather_tokens(pos, context_indices)
        context = self.context_proj(context_tokens) + context_pos
        context = context.repeat(num_targets, 1, 1)

        flat_target_indices = target_indices.reshape(num_targets * batch_size, target_length)
        target_pos = gather_tokens(
            self.pos_embed.expand(num_targets * batch_size, -1, -1).to(dtype=context_tokens.dtype),
            flat_target_indices,
        )
        target = self.mask_token.expand(num_targets * batch_size, target_length, -1) + target_pos
        x = torch.cat([context, target], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.output_proj(self.norm(x[:, -target_length:]))
        return x.reshape(num_targets, batch_size, target_length, -1)


class IJEPAModel(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        encoder_dim: int,
        encoder_depth: int,
        encoder_heads: int,
        predictor_dim: int,
        predictor_depth: int,
        predictor_heads: int,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.context_encoder = VisionTransformer(
            image_size, patch_size, encoder_dim, encoder_depth, encoder_heads, mlp_ratio
        )
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.target_encoder.requires_grad_(False)
        self.predictor = Predictor(
            image_size // patch_size,
            encoder_dim,
            predictor_dim,
            predictor_depth,
            predictor_heads,
            mlp_ratio,
        )

    @torch.no_grad()
    def targets(self, images: torch.Tensor, target_indices: torch.Tensor, normalize: bool) -> torch.Tensor:
        full = self.target_encoder(images)
        if normalize:
            full = F.layer_norm(full, (full.size(-1),))
        return torch.stack([gather_tokens(full, indices) for indices in target_indices])

    def predictions(self, images: torch.Tensor, masks: MaskBatch) -> torch.Tensor:
        context = self.context_encoder(images, masks.context)
        return self.predictor(context, masks.context, masks.targets)

    def forward(self, images: torch.Tensor, masks: MaskBatch, normalize_targets: bool = True):
        target = self.targets(images, masks.targets, normalize_targets)
        prediction = self.predictions(images, masks)
        return prediction, target

    @torch.no_grad()
    def ema_update(self, momentum: float) -> None:
        for context, target in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            target.mul_(momentum).add_(context.detach(), alpha=1.0 - momentum)

    def trainable_parameters(self):
        yield from self.context_encoder.parameters()
        yield from self.predictor.parameters()


def latent_loss(prediction: torch.Tensor, target: torch.Tensor, kind: str) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction {prediction.shape} != target {target.shape}")
    if kind == "l2":
        return (prediction - target).square().mean()
    if kind == "smooth_l1":
        return F.smooth_l1_loss(prediction, target)
    raise ValueError(f"unknown loss: {kind}")
