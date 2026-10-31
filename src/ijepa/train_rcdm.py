"""Train the latent RCDM visualizer on a precomputed (latent, feature) cache (decision D-011).

    python -m src.ijepa.train_rcdm --config configs/<rcdm>.yaml

Config sections: seed, cache (train/val paths), model (RCDMUNet kwargs), optimization,
training. The I-JEPA encoder and the VAE are never trained; only the UNet is. An EMA copy of
the UNet is kept for sampling. Checkpoints reuse train.save_checkpoint (atomic, resumable).
`optimization.ema_warmup: true` ramps the EMA decay as min(ema_decay, (1 + t) / (10 + t)), so
early sample grids are not dominated by the random init (0.9999^5000 = 0.61 of it without).
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import Config
from .rcdm import RCDMUNet, decode_latents, diffusion_loss, load_vae, make_scheduler, sample
from .train import cosine_schedule, load_checkpoint, save_checkpoint


class RCDMState(nn.Module):
    """Holds the trained UNet and its EMA so both live in one resumable state_dict."""

    def __init__(self, model_cfg: dict) -> None:
        super().__init__()
        self.model = RCDMUNet(**model_cfg)
        self.ema = copy.deepcopy(self.model).requires_grad_(False)

    @torch.no_grad()
    def update_ema(self, decay: float) -> None:
        for ema_p, p in zip(self.ema.parameters(), self.model.parameters()):
            ema_p.mul_(decay).add_(p.detach(), alpha=1 - decay)


def ema_decay_at(step: int, max_decay: float, warmup: bool) -> float:
    """EMA decay for the update after optimizer step `step` (0-based); depends only on the step, so resumes match."""
    return min(max_decay, (1 + step) / (10 + step)) if warmup else max_decay


def load_cache(path: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["latents"].size(0) != payload["features"].size(0):
        raise ValueError("cache latents and features are not aligned")
    return payload


@torch.no_grad()
def save_rcdm_grid(path: Path, ema: RCDMUNet, vae, val: dict, count: int, samples_per_image: int,
                   steps: int, device: torch.device, seed: int) -> None:
    """Grid: first row = VAE reconstruction of the real image; next rows = samples from its feature."""
    from torchvision.utils import save_image

    latents = val["latents"][:count].float().to(device)
    features = val["features"][:count].float().to(device)
    rows = [decode_latents(vae, latents).cpu()]
    generator = torch.Generator(device=device).manual_seed(seed)
    for _ in range(samples_per_image):
        rows.append(decode_latents(vae, sample(ema, features, steps=steps, generator=generator)).cpu())
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(torch.cat(rows), path, nrow=count)


def train_rcdm(config_path: str, commit_callback=None) -> dict:
    cfg = Config.load(config_path).raw
    seed = int(cfg.get("seed", 0))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(cfg["training"].get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.backends.cuda.matmul.allow_tf32 = True

    train_cache = load_cache(cfg["cache"]["train"])
    val_cache = load_cache(cfg["cache"]["val"]) if cfg["cache"].get("val") else train_cache
    feature_dim = int(train_cache["features"].shape[1])
    model_cfg = {**cfg.get("model", {}), "feature_dim": feature_dim,
                 "latent_size": int(train_cache["latents"].shape[-1])}
    state_module = RCDMState(model_cfg).to(device)
    scheduler = make_scheduler(int(cfg.get("diffusion", {}).get("num_train_timesteps", 1000)))

    opt_cfg, tr = cfg["optimization"], cfg["training"]
    optimizer = torch.optim.AdamW(state_module.model.parameters(), lr=float(opt_cfg["lr"]),
                                  weight_decay=float(opt_cfg.get("weight_decay", 0.0)))
    precision = tr.get("precision", "fp32")
    use_amp = device.type == "cuda" and precision == "bf16"
    scaler = torch.amp.GradScaler("cuda", enabled=False)  # bf16 needs no loss scaling

    total_steps, batch_size = int(tr["total_steps"]), int(tr["batch_size"])
    output = Path(tr["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    latest, log_path = output / "latest.pt", output / "metrics.jsonl"
    state = {"global_step": 0, "epoch": 0, "batch_in_epoch": 0}
    if tr.get("resume", True) and latest.exists():
        state = load_checkpoint(latest, state_module, optimizer, scaler, device)

    vae = None
    latents_all, features_all = train_cache["latents"], train_cache["features"]
    n = latents_all.size(0)
    generator = torch.Generator().manual_seed(seed + int(state["global_step"]))
    start_step, start_time = int(state["global_step"]), time.perf_counter()
    state_module.model.train()
    while state["global_step"] < total_steps:
        step = int(state["global_step"])
        index = torch.randint(0, n, (batch_size,), generator=generator)
        latents = latents_all[index].float().to(device, non_blocking=True)
        features = features_all[index].float().to(device, non_blocking=True)
        if float(opt_cfg.get("cond_drop_prob", 0.0)) > 0:  # optional, off by default (RCDM has none)
            keep = (torch.rand(batch_size, generator=generator) >= float(opt_cfg["cond_drop_prob"])).float()
            features = features * keep.to(device).unsqueeze(1)
        lr = cosine_schedule(float(opt_cfg["lr"]), float(opt_cfg.get("final_lr", opt_cfg["lr"])), step, total_steps,
                             int(opt_cfg.get("warmup_steps", 0)), float(opt_cfg.get("start_lr", 0.0)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            loss = diffusion_loss(state_module.model, scheduler, latents, features)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite RCDM loss at step {step}: {loss}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(state_module.model.parameters(), float(opt_cfg.get("grad_clip", 1.0)))
        optimizer.step()
        ema_decay = ema_decay_at(step, float(opt_cfg.get("ema_decay", 0.9999)), bool(opt_cfg.get("ema_warmup", False)))
        state_module.update_ema(ema_decay)
        state = {"global_step": step + 1, "epoch": (step + 1) * batch_size // n, "batch_in_epoch": 0}

        if step % int(tr["log_every"]) == 0 or step + 1 == total_steps:
            elapsed = max(time.perf_counter() - start_time, 1e-6)
            metrics = {"step": step + 1, "loss": float(loss), "lr": lr, "grad_norm": float(grad_norm), "ema_decay": ema_decay,
                       "images_per_second": (step + 1 - start_step) * batch_size / elapsed,
                       "peak_memory_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0}
            print(json.dumps(metrics), flush=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics) + "\n")

        sample_every = int(tr.get("sample_every", 0))
        if sample_every and ((step + 1) % sample_every == 0 or step + 1 == total_steps):
            if vae is None:
                vae = load_vae(device, cache_dir=cfg.get("vae_cache"))
            state_module.ema.eval()
            save_rcdm_grid(output / f"samples_step_{step + 1:07d}.png", state_module.ema, vae, val_cache,
                           int(tr.get("sample_images", 8)), int(tr.get("samples_per_image", 3)),
                           int(tr.get("sample_steps", 50)), device, seed)

        if (step + 1) % int(tr["checkpoint_every"]) == 0 or step + 1 == total_steps:
            save_checkpoint(latest, state_module, optimizer, scaler, state, {**cfg, "model": model_cfg})
            archive_every = int(tr.get("archive_every", 0))
            if archive_every and (step + 1) % archive_every == 0:
                save_checkpoint(output / f"step_{step + 1:07d}.pt", state_module, optimizer, scaler, state,
                                {**cfg, "model": model_cfg})
            if commit_callback is not None:
                commit_callback()
    return {"checkpoint": str(latest), "state": state, "feature_dim": feature_dim,
            "cache_meta": train_cache.get("meta")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    train_rcdm(parser.parse_args().config)


if __name__ == "__main__":
    main()
