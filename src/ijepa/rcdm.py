"""RCDM-style representation-conditioned diffusion, in the SD-VAE latent space (decision D-011).

RCDM (Bordes et al., arXiv:2112.09164) conditions an ADM UNet on a frozen SSL representation
by projecting it and feeding it to the normalization layers of every residual block. Here the
same idea runs on 4x28x28 Stable-Diffusion-VAE latents of 224 px images:

    feature h --MLP--> vector added to the timestep embedding --> scale/shift GroupNorm in
    every UNet residual block (diffusers ``resnet_time_scale_shift="scale_shift"``).

The I-JEPA model and the VAE are frozen; only the UNet (+ its conditioning MLP) is trained.
"""
from __future__ import annotations

import torch
from torch import nn

VAE_NAME = "stabilityai/sd-vae-ft-mse"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RCDMUNet(nn.Module):
    """UNet over VAE latents whose every residual block is modulated by an I-JEPA feature."""

    def __init__(
        self,
        feature_dim: int,
        latent_size: int = 28,
        latent_channels: int = 4,
        block_out_channels: tuple[int, ...] = (128, 256, 512),
        layers_per_block: int = 2,
        attention_levels: tuple[bool, ...] = (False, True, True),
        attention_head_dim: int = 64,
        cond_hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        from diffusers import UNet2DModel

        down = tuple("AttnDownBlock2D" if a else "DownBlock2D" for a in attention_levels)
        up = tuple("AttnUpBlock2D" if a else "UpBlock2D" for a in reversed(attention_levels))
        self.unet = UNet2DModel(
            sample_size=latent_size,
            in_channels=latent_channels,
            out_channels=latent_channels,
            block_out_channels=tuple(block_out_channels),
            layers_per_block=layers_per_block,
            down_block_types=down,
            up_block_types=up,
            attention_head_dim=attention_head_dim,
            class_embed_type="identity",  # conditioning vector is added to the time embedding
            resnet_time_scale_shift="scale_shift",  # ...which modulates every residual block
        )
        time_embed_dim = block_out_channels[0] * 4
        # RCDM: "a fully connected layer to h that reduces dimension to a vector of size 512".
        self.cond_proj = nn.Sequential(
            nn.Linear(feature_dim, cond_hidden_dim),
            nn.SiLU(),
            nn.Linear(cond_hidden_dim, time_embed_dim),
        )

    def forward(self, noisy_latents: torch.Tensor, timesteps: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        condition = self.cond_proj(features.to(self.cond_proj[0].weight.dtype))
        return self.unet(noisy_latents, timesteps, class_labels=condition).sample


def make_scheduler(num_train_timesteps: int = 1000):
    from diffusers import DDPMScheduler

    # Latent-diffusion default schedule; epsilon prediction.
    return DDPMScheduler(num_train_timesteps=num_train_timesteps, beta_schedule="scaled_linear",
                         beta_start=0.00085, beta_end=0.012, prediction_type="epsilon")


def load_vae(device: torch.device | str, dtype: torch.dtype = torch.float32, cache_dir: str | None = None):
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(VAE_NAME, cache_dir=cache_dir, torch_dtype=dtype)
    vae.requires_grad_(False)
    return vae.eval().to(device)


def imagenet_normalized_to_vae_range(images: torch.Tensor) -> torch.Tensor:
    """Our data pipeline yields ImageNet-normalized tensors; the SD-VAE expects [-1, 1]."""
    mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return ((images * std + mean).clamp(0, 1) * 2 - 1)


@torch.no_grad()
def encode_latents(vae, images_normalized: torch.Tensor) -> torch.Tensor:
    """Deterministic latent (posterior mean) scaled by the VAE scaling factor."""
    x = imagenet_normalized_to_vae_range(images_normalized).to(vae.dtype)
    return vae.encode(x).latent_dist.mean * vae.config.scaling_factor


@torch.no_grad()
def decode_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    """Scaled latents -> images in [0, 1]."""
    x = vae.decode((latents / vae.config.scaling_factor).to(vae.dtype)).sample
    return ((x.float() + 1) / 2).clamp(0, 1)


@torch.no_grad()
def pooled_target_features(ijepa_model, images_normalized: torch.Tensor) -> torch.Tensor:
    """Average-pooled target-encoder output: the representation k-NN evaluation uses."""
    return ijepa_model.target_encoder(images_normalized).mean(dim=1)


def diffusion_loss(model: RCDMUNet, scheduler, latents: torch.Tensor, features: torch.Tensor,
                   generator: torch.Generator | None = None) -> torch.Tensor:
    noise = torch.randn(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)
    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.size(0),),
                              device=latents.device, generator=generator)
    noisy = scheduler.add_noise(latents, noise, timesteps)
    prediction = model(noisy, timesteps, features)
    return torch.nn.functional.mse_loss(prediction.float(), noise.float())


def _ddim(steps: int):
    from diffusers import DDIMScheduler

    ddim = DDIMScheduler(num_train_timesteps=1000, beta_schedule="scaled_linear",
                         beta_start=0.00085, beta_end=0.012, prediction_type="epsilon",
                         clip_sample=False, set_alpha_to_one=False)
    ddim.set_timesteps(steps)
    return ddim


@torch.no_grad()
def sample(model: RCDMUNet, features: torch.Tensor, steps: int = 50, generator: torch.Generator | None = None,
           latent_shape: tuple[int, int, int] = (4, 28, 28), eta: float = 0.0) -> torch.Tensor:
    """DDIM sampling of latents conditioned on `features` [N, D]."""
    ddim = _ddim(steps)
    device = features.device
    latents = torch.randn((features.size(0), *latent_shape), generator=generator, device=device)
    for t in ddim.timesteps:
        eps = model(latents, t.expand(latents.size(0)).to(device), features)
        latents = ddim.step(eps.float(), t, latents, eta=eta, generator=generator).prev_sample
    return latents


@torch.no_grad()
def inpaint(model: RCDMUNet, features: torch.Tensor, known: torch.Tensor, keep: torch.Tensor, steps: int = 50,
            generator: torch.Generator | None = None, eta: float = 0.0, resample: int = 1) -> torch.Tensor:
    """DDIM sampling that generates only the latent cells with keep == 0.

    After every step the cells with keep == 1 are reset to `known` noised to the next step's level, and at
    the end to `known` itself (blended latent diffusion; the UNet is not retrained for inpainting).
    known: [N,4,h,w] scaled latents; keep: [N,1,h,w] (or broadcastable) in {0, 1}.
    resample > 1 repeats every step that many times, re-noising the blended result back to the current level in
    between (RePaint, Lugmayr et al. 2022), so the generated cells can adapt to the known ones.
    """
    ddim = _ddim(steps)
    device = features.device
    keep = keep.to(device=device, dtype=known.dtype)
    stride = ddim.config.num_train_timesteps // ddim.num_inference_steps
    latents = torch.randn(known.shape, generator=generator, device=device)
    alphas = ddim.alphas_cumprod.to(device)
    for t in ddim.timesteps:
        previous = int(t) - stride
        current = latents
        for repeat in range(resample):
            eps = model(current, t.expand(current.size(0)).to(device), features)
            latents = ddim.step(eps.float(), t, current, eta=eta, generator=generator).prev_sample
            if previous >= 0:
                noise = torch.randn(known.shape, generator=generator, device=device)
                target = ddim.add_noise(known, noise, torch.full((known.size(0),), previous, device=device))
            else:
                target = known
            latents = keep * target + (1 - keep) * latents
            if repeat + 1 < resample and previous >= 0:  # forward-diffuse previous -> t and redo the step
                ratio = alphas[int(t)] / alphas[previous]
                noise = torch.randn(known.shape, generator=generator, device=device)
                current = ratio.sqrt() * latents + (1 - ratio).sqrt() * noise
    return latents

