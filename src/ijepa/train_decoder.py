from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .data import make_dataset, make_loader
from .decoder import (
    MaskedPatchDecoder,
    compose_masked_images,
    gather_target_patches,
    patchify,
)
from .masks import MaskCollator, MultiBlockMaskSampler
from .model import IJEPAModel
from .train import cosine_schedule, load_checkpoint, save_checkpoint

# Split that make_dataset(train=True) reads, re-opened with the eval transform for held-out scoring.
_TRAIN_SPLIT = {"stl10": "unlabeled", "imagenet100": "train"}


def split_holdout(count: int, holdout_size: int, seed: int) -> tuple[list[int], list[int]]:
    """Deterministic (train_indices, holdout_indices) partition of range(count)."""
    if not 0 < holdout_size < count:
        raise ValueError(f"holdout_size must be in (0, {count}); got {holdout_size}")
    perm = torch.randperm(count, generator=torch.Generator().manual_seed(seed)).tolist()
    return sorted(perm[holdout_size:]), sorted(perm[:holdout_size])


@torch.no_grad()
def holdout_metrics(encoder, decoder, loader, mask_sampler, conditioning, normalize_targets,
                    patch_size, device, use_amp, amp_dtype, mask_seed) -> dict:
    """Masked L1/MSE over a fixed held-out set with fixed masks (same masks at every evaluation)."""
    was_training = decoder.training
    decoder.eval()
    generator = torch.Generator().manual_seed(mask_seed)
    abs_sum = sq_sum = elements = 0.0
    for images, _labels in loader:
        images = images.to(device, non_blocking=True)
        masks = mask_sampler.sample(images.size(0), generator).to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            tokens = decoder_conditioning(encoder, images, masks, conditioning, normalize_targets)
            predicted = decoder(tokens, masks.targets)
        target = gather_target_patches(patchify(images, patch_size), masks.targets)
        diff = predicted.float() - target.float()
        abs_sum += diff.abs().sum().item()
        sq_sum += diff.pow(2).sum().item()
        elements += diff.numel()
    decoder.train(was_training)
    return {"val_masked_l1": abs_sum / elements, "val_masked_mse": sq_sum / elements}


def decoder_parameter_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if parameter.ndim == 1 or name.endswith("bias") else decay).append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def load_frozen_ijepa(path: str | Path, device: torch.device) -> tuple[IJEPAModel, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = IJEPAModel(**checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["model"])
    model.requires_grad_(False)
    model.eval()
    return model.to(device), checkpoint


CONDITIONING_MODES = ("predicted", "oracle")


@torch.no_grad()
def decoder_conditioning(
    encoder: IJEPAModel, images: torch.Tensor, masks, mode: str, normalize: bool
) -> torch.Tensor:
    """Latent tokens fed to the decoder, shaped [M,B,Nt,D].

    "predicted" is the deployable path (context encoder + predictor). "oracle" uses the
    target-encoder tokens the predictor was trained to match; it is an upper-bound
    diagnostic and cannot be used for real completion.
    """
    if mode == "predicted":
        return encoder.predictions(images, masks)
    if mode == "oracle":
        return encoder.targets(images, masks.targets, normalize)
    raise ValueError(f"unknown conditioning: {mode}; expected one of {CONDITIONING_MODES}")


def denormalize(images: torch.Tensor) -> torch.Tensor:
    mean = images.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = images.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0, 1)


@torch.no_grad()
def save_sample_grid(
    path: Path,
    images: torch.Tensor,
    predicted_patches: torch.Tensor,
    target_indices: torch.Tensor,
    patch_size: int,
    max_samples: int = 8,
) -> None:
    from torchvision.utils import save_image

    count = min(max_samples, images.size(0))
    images = images[:count]
    predicted_patches = predicted_patches[:, :count]
    target_indices = target_indices[:, :count]
    masked, completed, _ = compose_masked_images(
        images, predicted_patches, target_indices, patch_size
    )
    grid = torch.cat(
        [denormalize(images), denormalize(masked), denormalize(completed)], dim=0
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, path, nrow=count)


def train_decoder(config_path: str, commit_callback=None) -> dict:
    cfg = Config.load(config_path).raw
    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True

    encoder, encoder_checkpoint = load_frozen_ijepa(cfg["encoder_checkpoint"], device)
    encoder_cfg = encoder_checkpoint["config"]
    model_cfg = encoder_cfg["model"]
    decoder = MaskedPatchDecoder(
        image_size=int(model_cfg["image_size"]),
        patch_size=int(model_cfg["patch_size"]),
        encoder_dim=int(model_cfg["encoder_dim"]),
        **cfg["decoder"],
    ).to(device)
    decoder_init_checkpoint = cfg.get("decoder_init_checkpoint")
    if decoder_init_checkpoint:
        init_payload = torch.load(
            decoder_init_checkpoint, map_location="cpu", weights_only=False
        )
        decoder.load_state_dict(init_payload["model"])
    mask_cfg = {
        **cfg.get("mask", encoder_cfg["mask"]),
        "image_size": int(model_cfg["image_size"]),
        "patch_size": int(model_cfg["patch_size"]),
    }
    mask_sampler = MultiBlockMaskSampler(**mask_cfg)
    conditioning = cfg.get("conditioning", "predicted")
    if conditioning not in CONDITIONING_MODES:
        raise ValueError(f"unknown conditioning: {conditioning}")
    normalize_targets = bool(encoder_cfg["objective"].get("normalize_targets", True))

    opt_cfg = cfg["optimization"]
    optimizer = torch.optim.AdamW(
        decoder_parameter_groups(decoder, float(opt_cfg["weight_decay"])),
        lr=float(opt_cfg["lr"]),
    )
    precision = cfg["training"].get("precision", "fp32")
    use_amp = device.type == "cuda" and precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and precision == "fp16")

    data_cfg = {**encoder_cfg["data"], **cfg.get("data", {})}
    dataset = make_dataset(data_cfg, train=True)
    tr = cfg["training"]
    # Optional held-out slice of the *training* data for model selection / early stopping; the
    # dataset's own validation split stays untouched as the benchmark.
    holdout_size = int(tr.get("holdout_size", 0))
    holdout_loader = None
    if holdout_size:
        train_idx, hold_idx = split_holdout(len(dataset), holdout_size, int(tr.get("holdout_seed", 0)))
        name = data_cfg["name"]
        eval_base = dataset if name == "synthetic" else make_dataset(data_cfg, train=False, split=_TRAIN_SPLIT[name])
        if len(eval_base) != len(dataset):
            raise ValueError("held-out base split does not match the training split")
        holdout_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(eval_base, hold_idx), batch_size=int(tr.get("eval_batch_size", data_cfg["batch_size"])),
            shuffle=False, num_workers=int(data_cfg.get("num_workers", 0)), pin_memory=bool(data_cfg.get("pin_memory", True)))
        dataset = torch.utils.data.Subset(dataset, train_idx)
    eval_every = int(tr.get("eval_every", 0)) if holdout_loader is not None else 0
    patience = int(tr.get("early_stop_patience", 0))
    eval_mask_seed = int(tr.get("eval_mask_seed", 20260831))
    masks_in_loader = bool(data_cfg.get("masks_in_loader", False))
    collate_fn = MaskCollator(mask_sampler) if masks_in_loader else None
    total_steps = int(tr["total_steps"])
    output = Path(tr["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    latest = output / "latest.pt"
    best_path = output / "best.pt"
    log_path = output / "metrics.jsonl"
    state = {"global_step": 0, "epoch": 0, "batch_in_epoch": 0}
    if tr.get("resume", True) and latest.exists():
        state = load_checkpoint(latest, decoder, optimizer, scaler, device)
    best_val = float(state.get("best_val_masked_l1", float("inf")))
    evals_since_best = int(state.get("evals_since_best", 0))

    def checkpoint_state() -> dict:
        return {**state, "best_val_masked_l1": best_val, "evals_since_best": evals_since_best}

    decoder.train()
    start_step = int(state["global_step"])
    start_time = time.perf_counter()
    mask_generator = torch.Generator().manual_seed(seed + start_step)
    patch_size = int(model_cfg["patch_size"])
    while state["global_step"] < total_steps:
        loader = make_loader(dataset, data_cfg, state["epoch"], collate_fn=collate_fn)
        for batch_index, batch in enumerate(loader):
            if batch_index < state["batch_in_epoch"]:
                continue
            images = batch[0]
            step = int(state["global_step"])
            lr = cosine_schedule(
                float(opt_cfg["lr"]),
                float(opt_cfg["final_lr"]),
                step,
                total_steps,
                int(opt_cfg.get("warmup_steps", 0)),
                float(opt_cfg.get("start_lr", 0.0)),
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            images = images.to(device, non_blocking=True)
            masks = (batch[2] if masks_in_loader else mask_sampler.sample(images.size(0), mask_generator)).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                predicted_tokens = decoder_conditioning(
                    encoder, images, masks, conditioning, normalize_targets
                )
                predicted_patches = decoder(predicted_tokens, masks.targets)
                target_patches = gather_target_patches(patchify(images, patch_size), masks.targets)
                loss = F.l1_loss(predicted_patches, target_patches)
                mse = F.mse_loss(predicted_patches, target_patches)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite decoder loss at step {step}: {loss}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                decoder.parameters(), float(opt_cfg.get("grad_clip", 1.0))
            )
            scaler.step(optimizer)
            scaler.update()
            state = {
                "global_step": step + 1,
                "epoch": state["epoch"],
                "batch_in_epoch": batch_index + 1,
            }

            if step % int(cfg["training"]["log_every"]) == 0 or step + 1 == total_steps:
                elapsed = max(time.perf_counter() - start_time, 1e-6)
                processed = (step + 1 - start_step) * images.size(0)
                metrics = {
                    "step": step + 1,
                    "epoch": state["epoch"],
                    "masked_l1": float(loss),
                    "masked_mse": float(mse),
                    "lr": lr,
                    "grad_norm": float(grad_norm),
                    "images_per_second": processed / elapsed,
                    "peak_memory_gb": (
                        torch.cuda.max_memory_allocated() / 2**30
                        if device.type == "cuda"
                        else 0.0
                    ),
                }
                print(json.dumps(metrics), flush=True)
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metrics) + "\n")

            sample_every = int(cfg["training"].get("sample_every", 0))
            if sample_every and (step + 1) % sample_every == 0:
                save_sample_grid(
                    output / f"samples_step_{step + 1:07d}.png",
                    images,
                    predicted_patches,
                    masks.targets,
                    patch_size,
                )

            stop_early = False
            if eval_every and ((step + 1) % eval_every == 0 or step + 1 == total_steps):
                val = holdout_metrics(encoder, decoder, holdout_loader, mask_sampler, conditioning,
                                      normalize_targets, patch_size, device, use_amp, amp_dtype, eval_mask_seed)
                improved = val["val_masked_l1"] < best_val
                if improved:
                    best_val, evals_since_best = val["val_masked_l1"], 0
                    save_checkpoint(best_path, decoder, optimizer, scaler, checkpoint_state(), cfg)
                else:
                    evals_since_best += 1
                row = {"step": step + 1, **val, "best_val_masked_l1": best_val, "improved": improved,
                       "evals_since_best": evals_since_best, "holdout_images": holdout_size}
                print(json.dumps(row), flush=True)
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                stop_early = bool(patience) and evals_since_best >= patience

            if (
                (step + 1) % int(tr["checkpoint_every"]) == 0
                or step + 1 == total_steps
                or stop_early
            ):
                save_checkpoint(latest, decoder, optimizer, scaler, checkpoint_state(), cfg)
                archive_every = int(tr.get("archive_every", 0))
                if archive_every and (step + 1) % archive_every == 0:
                    save_checkpoint(
                        output / f"step_{step + 1:07d}.pt",
                        decoder,
                        optimizer,
                        scaler,
                        checkpoint_state(),
                        cfg,
                    )
                if commit_callback is not None:
                    commit_callback()
            if stop_early:
                print(json.dumps({"early_stopped": True, "step": step + 1, "best_val_masked_l1": best_val}), flush=True)
                return {"checkpoint": str(latest), "best_checkpoint": str(best_path), "state": checkpoint_state(),
                        "early_stopped": True, "conditioning": conditioning,
                        "encoder_checkpoint": str(cfg["encoder_checkpoint"]),
                        "encoder_step": int(encoder_checkpoint["state"]["global_step"])}
            if state["global_step"] >= total_steps:
                break
        if state["batch_in_epoch"] >= len(loader):
            state["epoch"] += 1
            state["batch_in_epoch"] = 0

    return {
        "checkpoint": str(latest),
        "best_checkpoint": str(best_path) if eval_every else None,
        "state": checkpoint_state(),
        "conditioning": conditioning,
        "encoder_checkpoint": str(cfg["encoder_checkpoint"]),
        "encoder_step": int(encoder_checkpoint["state"]["global_step"]),
        "decoder_init_checkpoint": (
            str(decoder_init_checkpoint) if decoder_init_checkpoint else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train_decoder(args.config)


if __name__ == "__main__":
    main()
