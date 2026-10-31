from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MaskBatch:
    context: torch.Tensor  # [B, Nc]
    targets: torch.Tensor  # [M, B, Nt]

    def to(self, device: torch.device | str) -> "MaskBatch":
        return MaskBatch(self.context.to(device), self.targets.to(device))


class MultiBlockMaskSampler:
    """Samples one context block and multiple target blocks on a patch grid.

    Block shapes are shared across a batch while locations vary per image. Target
    blocks may overlap each other; their union is always removed from the context.
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_targets: int = 4,
        target_scale: tuple[float, float] = (0.15, 0.20),
        context_scale: tuple[float, float] = (0.85, 1.00),
        target_aspect: tuple[float, float] = (0.75, 1.50),
        min_context_patches: int = 10,
    ) -> None:
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.grid_size = image_size // patch_size
        self.num_targets = num_targets
        self.target_scale = target_scale
        self.context_scale = context_scale
        self.target_aspect = target_aspect
        self.min_context_patches = min_context_patches

    @property
    def num_patches(self) -> int:
        return self.grid_size**2

    def _shape(
        self,
        scale: tuple[float, float],
        aspect: tuple[float, float],
        generator: torch.Generator,
    ) -> tuple[int, int]:
        area_fraction = torch.empty(1).uniform_(*scale, generator=generator).item()
        ratio = torch.empty(1).uniform_(*aspect, generator=generator).item()
        area = max(1, round(self.num_patches * area_fraction))
        height = max(1, min(self.grid_size, round(math.sqrt(area * ratio))))
        width = max(1, min(self.grid_size, round(math.sqrt(area / ratio))))
        return height, width

    def _block(
        self,
        height: int,
        width: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        top = int(torch.randint(self.grid_size - height + 1, (1,), generator=generator))
        left = int(torch.randint(self.grid_size - width + 1, (1,), generator=generator))
        rows = torch.arange(top, top + height)
        cols = torch.arange(left, left + width)
        return (rows[:, None] * self.grid_size + cols[None, :]).flatten()

    def sample(self, batch_size: int, generator: torch.Generator | None = None) -> MaskBatch:
        generator = generator or torch.default_generator
        target_h, target_w = self._shape(self.target_scale, self.target_aspect, generator)
        context_h, context_w = self._shape(self.context_scale, (1.0, 1.0), generator)

        all_targets: list[list[torch.Tensor]] = [[] for _ in range(self.num_targets)]
        contexts: list[torch.Tensor] = []
        for _ in range(batch_size):
            sample_targets = [self._block(target_h, target_w, generator) for _ in range(self.num_targets)]
            for target_id, target in enumerate(sample_targets):
                all_targets[target_id].append(target)
            target_union = torch.unique(torch.cat(sample_targets))

            context = torch.empty(0, dtype=torch.long)
            for _attempt in range(20):
                candidate = self._block(context_h, context_w, generator)
                keep = ~torch.isin(candidate, target_union)
                context = candidate[keep]
                if context.numel() >= self.min_context_patches:
                    break
            if context.numel() < self.min_context_patches:
                raise RuntimeError("could not sample a sufficiently large context block")
            contexts.append(torch.sort(context).values)

        # Removing target overlap makes context lengths vary. Match the official
        # batching constraint by using the smallest length on this device.
        context_length = min(x.numel() for x in contexts)
        fixed_contexts = []
        for context in contexts:
            if context.numel() > context_length:
                order = torch.randperm(context.numel(), generator=generator)[:context_length]
                context = torch.sort(context[order]).values
            fixed_contexts.append(context)

        return MaskBatch(
            context=torch.stack(fixed_contexts),
            targets=torch.stack([torch.stack(targets) for targets in all_targets]),
        )


class MaskCollator:
    """DataLoader collate_fn that samples masks inside the loader workers.

    Mirrors the official design (Assran et al. 2023, Appendix A.1: "a batch-collator function, which runs in
    the dataloader processes"), so mask sampling overlaps GPU work instead of running serially
    in the training loop. Each worker lazily seeds its own generator from torch.initial_seed(),
    which DataLoader sets per worker from the loader's (seed + epoch) generator.
    """

    def __init__(self, sampler: MultiBlockMaskSampler) -> None:
        self.sampler = sampler
        self._generator: torch.Generator | None = None

    def __call__(self, batch):
        if self._generator is None:
            self._generator = torch.Generator().manual_seed(torch.initial_seed() % (2**63))
        images = torch.stack([item[0] for item in batch])
        labels = torch.as_tensor([item[1] for item in batch])
        return images, labels, self.sampler.sample(images.size(0), self._generator)
