from __future__ import annotations

import torch
from torch import nn

from .model import Block, gather_tokens, sincos_2d


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    if images.ndim != 4:
        raise ValueError("expected images [B,C,H,W]")
    batch, channels, height, width = images.shape
    if height != width or height % patch_size:
        raise ValueError("images must be square and divisible by patch_size")
    grid = height // patch_size
    patches = images.reshape(batch, channels, grid, patch_size, grid, patch_size)
    patches = patches.permute(0, 2, 4, 1, 3, 5)
    return patches.reshape(batch, grid * grid, channels * patch_size * patch_size)


def unpatchify(
    patches: torch.Tensor,
    patch_size: int,
    image_size: int,
    channels: int = 3,
) -> torch.Tensor:
    if patches.ndim != 3:
        raise ValueError("expected patches [B,N,P]")
    grid = image_size // patch_size
    expected = channels * patch_size * patch_size
    if patches.size(1) != grid * grid or patches.size(2) != expected:
        raise ValueError("patch tensor does not match requested image geometry")
    images = patches.reshape(
        patches.size(0), grid, grid, channels, patch_size, patch_size
    )
    images = images.permute(0, 3, 1, 4, 2, 5)
    return images.reshape(patches.size(0), channels, image_size, image_size)


def gather_target_patches(
    image_patches: torch.Tensor,
    target_indices: torch.Tensor,
) -> torch.Tensor:
    if target_indices.ndim != 3:
        raise ValueError("expected target_indices [M,B,Nt]")
    return torch.stack(
        [gather_tokens(image_patches, indices) for indices in target_indices]
    )


def compose_masked_images(
    images: torch.Tensor,
    predicted_patches: torch.Tensor,
    target_indices: torch.Tensor,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return masked images, completions, and a per-patch target mask.

    Predictions for target blocks that overlap are averaged. Non-target patches
    in the completion are copied bit-for-bit from the input patch tensor.
    """
    original = patchify(images, patch_size)
    if predicted_patches.shape[:3] != target_indices.shape:
        raise ValueError("predictions and target indices do not align")
    if predicted_patches.size(-1) != original.size(-1):
        raise ValueError("predicted patch dimension does not match images")

    accumulated = torch.zeros_like(original)
    counts = torch.zeros(
        original.size(0), original.size(1), 1, device=original.device, dtype=original.dtype
    )
    for block_predictions, block_indices in zip(predicted_patches, target_indices):
        block_predictions = block_predictions.to(dtype=original.dtype)
        expanded_indices = block_indices.unsqueeze(-1).expand_as(block_predictions)
        accumulated.scatter_add_(1, expanded_indices, block_predictions)
        counts.scatter_add_(
            1,
            block_indices.unsqueeze(-1),
            torch.ones_like(block_indices, dtype=original.dtype).unsqueeze(-1),
        )

    target_mask = counts > 0
    averaged = accumulated / counts.clamp_min(1)
    completed_patches = torch.where(target_mask, averaged, original)
    masked_patches = torch.where(target_mask, torch.zeros_like(original), original)
    image_size = images.size(-1)
    return (
        unpatchify(masked_patches, patch_size, image_size, images.size(1)),
        unpatchify(completed_patches, patch_size, image_size, images.size(1)),
        target_mask.squeeze(-1),
    )


class MaskedPatchDecoder(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        encoder_dim: int,
        decoder_dim: int = 256,
        depth: int = 4,
        heads: int = 8,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.input_proj = nn.Linear(encoder_dim, decoder_dim)
        self.register_buffer(
            "pos_embed", sincos_2d(self.grid_size, decoder_dim), persistent=True
        )
        self.blocks = nn.ModuleList(
            [Block(decoder_dim, heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(decoder_dim)
        self.patch_head = nn.Linear(decoder_dim, 3 * patch_size * patch_size)
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        predicted_tokens: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        if predicted_tokens.ndim != 4 or target_indices.ndim != 3:
            raise ValueError("expected tokens [M,B,Nt,D] and indices [M,B,Nt]")
        if predicted_tokens.shape[:3] != target_indices.shape:
            raise ValueError("predicted tokens and target indices do not align")

        num_targets, batch_size, target_length, _ = predicted_tokens.shape
        flat = predicted_tokens.reshape(num_targets * batch_size, target_length, -1)
        flat_indices = target_indices.reshape(num_targets * batch_size, target_length)
        positions = gather_tokens(
            self.pos_embed.expand(num_targets * batch_size, -1, -1).to(flat.dtype),
            flat_indices,
        )
        x = self.input_proj(flat) + positions
        for block in self.blocks:
            x = block(x)
        patches = self.patch_head(self.norm(x))
        return patches.reshape(num_targets, batch_size, target_length, -1)
