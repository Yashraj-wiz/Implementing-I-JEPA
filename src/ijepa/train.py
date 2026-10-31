from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .data import make_dataset, make_loader
from .masks import MaskCollator, MultiBlockMaskSampler
from .model import IJEPAModel, latent_loss


def interpolate(start: float, end: float, step: int, total: int) -> float:
    return start + (end - start) * min(step / max(total - 1, 1), 1.0)


def cosine_schedule(start: float, end: float, step: int, total: int, warmup: int = 0, warmup_start: float = 0.0):
    if warmup and step < warmup:
        return interpolate(warmup_start, start, step, warmup)
    progress = min(max((step - warmup) / max(total - warmup - 1, 1), 0.0), 1.0)
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))


def parameter_groups(model: IJEPAModel, weight_decay: float):
    decay, no_decay = [], []
    for name, parameter in list(model.context_encoder.named_parameters()) + list(model.predictor.named_parameters()):
        (no_decay if parameter.ndim == 1 or name.endswith("bias") else decay).append(parameter)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def feature_stats(target: torch.Tensor) -> tuple[float, float]:
    flat = target.detach().float().permute(1, 0, 2, 3).reshape(target.size(1), -1, target.size(-1)).mean(1)
    std = flat.std(dim=0).mean().item()
    normalized = torch.nn.functional.normalize(flat, dim=-1)
    similarity = normalized @ normalized.T
    if flat.size(0) > 1:
        mean_cos = ((similarity.sum() - similarity.diag().sum()) / (flat.size(0) * (flat.size(0) - 1))).item()
    else:
        mean_cos = 1.0
    return std, mean_cos


def save_checkpoint(path: Path, model, optimizer, scaler, state: dict, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        "state": state, "config": config,
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: Path, model, optimizer, scaler, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"]); optimizer.load_state_dict(payload["optimizer"])
    if scaler.is_enabled() and payload["scaler"] is not None: scaler.load_state_dict(payload["scaler"])
    random.setstate(payload["rng"]["python"]); np.random.set_state(payload["rng"]["numpy"])
    torch.set_rng_state(payload["rng"]["torch"].cpu())
    if torch.cuda.is_available() and payload["rng"]["cuda"] is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in payload["rng"]["cuda"]])
    return payload["state"]


def prepare_resume_state(state: dict, reset_data_cursor: bool) -> dict:
    state = dict(state)
    if reset_data_cursor:
        state["batch_in_epoch"] = 0
    return state


def train(config_path: str, commit_callback=None) -> dict:
    cfg = Config.load(config_path).raw
    seed = int(cfg.get("seed", 0)); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True

    model = IJEPAModel(**cfg["model"]).to(device)
    mask_cfg = {**cfg["mask"], "image_size": cfg["model"]["image_size"], "patch_size": cfg["model"]["patch_size"]}
    mask_sampler = MultiBlockMaskSampler(**mask_cfg)
    opt_cfg = cfg["optimization"]
    optimizer = torch.optim.AdamW(parameter_groups(model, float(opt_cfg["weight_decay"])), lr=float(opt_cfg["lr"]))
    precision = cfg["training"].get("precision", "fp32")
    use_amp = device.type == "cuda" and precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and precision == "fp16")

    dataset = make_dataset(cfg["data"], train=True)
    grad_accum_steps = max(1, int(cfg["training"].get("grad_accum_steps", 1)))  # micro-batches per optimizer step
    steps_per_epoch = len(dataset) // int(cfg["data"]["batch_size"]) // grad_accum_steps
    total_steps = int(cfg["training"].get("total_steps") or steps_per_epoch * int(cfg["training"]["epochs"]))
    output = Path(cfg["training"]["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    latest = output / "latest.pt"; log_path = output / "metrics.jsonl"
    state = {"global_step": 0, "epoch": 0, "batch_in_epoch": 0}
    if cfg["training"].get("resume", True) and latest.exists():
        state = load_checkpoint(latest, model, optimizer, scaler, device)
        state = prepare_resume_state(
            state,
            bool(cfg["training"].get("reset_data_cursor_on_resume", False)),
        )

    model.train(); start_step = int(state["global_step"]); start_time = time.perf_counter(); mask_generator = torch.Generator().manual_seed(seed + state["global_step"])
    # Opt-in speedups (defaults keep older configs' behaviour): masks sampled in loader workers,
    # and GPU->CPU syncs (finite-loss check, feature stats) only on log/checkpoint steps.
    masks_in_loader = bool(cfg["data"].get("masks_in_loader", False))
    sync_every_step = bool(cfg["training"].get("sync_every_step", True))
    collate_fn = MaskCollator(mask_sampler) if masks_in_loader else None
    micro_step = 0  # micro-batches accumulated toward the current optimizer step
    last_epoch_saved = int(state["epoch"]) - 1 if cfg["training"].get("epoch_checkpoint_from") is not None else -1
    if cfg["training"].get("epoch_checkpoint_from") is not None and int(state["epoch"]) < int(cfg["training"]["epoch_checkpoint_from"]):
        last_epoch_saved = -1
    epoch_from = cfg["training"].get("epoch_checkpoint_from")
    epoch_every = max(1, int(cfg["training"].get("epoch_checkpoint_every", 1)))
    dense_from = cfg["training"].get("epoch_checkpoint_dense_from")  # snapshot every epoch from this epoch on

    def write_epoch_snapshot(epoch_number: int, refresh_latest: bool = True) -> None:
        """bf16 weights `epoch_NNN_weights.pt` (+ immutable full `epoch_NNN.pt` with epoch_checkpoint_full) and
        a refreshed rolling latest.pt. Existing snapshots are never overwritten (a resume can land inside an
        epoch that was already saved)."""
        tag = f"epoch_{epoch_number:03d}"
        weights_path, full_path = output / f"{tag}_weights.pt", output / f"{tag}.pt"
        write_weights = not weights_path.exists()
        write_full = bool(cfg["training"].get("epoch_checkpoint_full", False)) and not full_path.exists()
        if not (write_weights or write_full):
            return
        if refresh_latest:
            save_checkpoint(latest, model, optimizer, scaler, state, cfg)
        if write_full:
            save_checkpoint(full_path, model, optimizer, scaler, state, cfg)
        if write_weights:
            weights = {"model": {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in model.state_dict().items()},
                       "state": state, "config": cfg, "note": "weights only (bf16); resume from latest.pt"}
            temporary = weights_path.with_suffix(".pt.tmp")
            torch.save(weights, temporary); os.replace(temporary, weights_path)
        if commit_callback is not None: commit_callback()

    while state["global_step"] < total_steps:
        loader = make_loader(dataset, cfg["data"], state["epoch"], collate_fn=collate_fn)
        for batch_index, batch in enumerate(loader):
            if batch_index < state["batch_in_epoch"]: continue
            images = batch[0]
            step = state["global_step"]
            lr = cosine_schedule(float(opt_cfg["lr"]), float(opt_cfg["final_lr"]), step, total_steps,
                                 int(opt_cfg["warmup_steps"]), float(opt_cfg["start_lr"]))
            wd = interpolate(float(opt_cfg["weight_decay"]), float(opt_cfg["final_weight_decay"]), step, total_steps)
            momentum = interpolate(float(opt_cfg["ema"][0]), float(opt_cfg["ema"][1]), step, total_steps)
            optimizer.param_groups[0]["lr"] = lr; optimizer.param_groups[1]["lr"] = lr
            optimizer.param_groups[0]["weight_decay"] = wd
            images = images.to(device, non_blocking=True)
            masks = (batch[2] if masks_in_loader else mask_sampler.sample(images.size(0), mask_generator)).to(device)
            if micro_step == 0:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                prediction, target = model(images, masks, bool(cfg["objective"].get("normalize_targets", True)))
                loss = latent_loss(prediction, target, cfg["objective"]["loss"])
            log_step = step % int(cfg["training"]["log_every"]) == 0 or step + 1 == total_steps
            checkpoint_step = (step + 1) % int(cfg["training"]["checkpoint_every"]) == 0 or step + 1 == total_steps
            if (sync_every_step or log_step or checkpoint_step) and not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}: {loss}")
            scaler.scale(loss / grad_accum_steps).backward()
            micro_step += 1
            if micro_step < grad_accum_steps:  # keep accumulating; the optimizer step comes later
                state = {"global_step": step, "epoch": state["epoch"], "batch_in_epoch": batch_index + 1}
                continue
            micro_step = 0
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), float(opt_cfg.get("grad_clip", 1.0)))
            scaler.step(optimizer); scaler.update(); model.ema_update(momentum)
            state = {"global_step": step + 1, "epoch": state["epoch"], "batch_in_epoch": batch_index + 1}

            if log_step:
                feature_std, mean_cos = feature_stats(target)
                elapsed = max(time.perf_counter() - start_time, 1e-6)
                processed = (step + 1 - start_step) * images.size(0) * grad_accum_steps
                metrics = {"step": step + 1, "epoch": state["epoch"], "loss": float(loss), "lr": lr, "wd": wd,
                           "ema": momentum, "grad_norm": float(grad_norm), "feature_std": feature_std,
                           "mean_cosine": mean_cos, "images_per_second": processed / elapsed,
                           "peak_memory_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0}
                print(json.dumps(metrics), flush=True)
                with open(log_path, "a", encoding="utf-8") as handle: handle.write(json.dumps(metrics) + "\n")

            if checkpoint_step:
                save_checkpoint(latest, model, optimizer, scaler, state, cfg)
                archive_every = int(cfg["training"].get("archive_every", 0))
                if archive_every and (step + 1) % archive_every == 0:
                    save_checkpoint(output / f"step_{step + 1:07d}.pt", model, optimizer, scaler, state, cfg)
                if commit_callback is not None: commit_callback()
            if (epoch_from is not None and state["epoch"] >= int(epoch_from) and state["epoch"] > last_epoch_saved
                    and ((state["epoch"] - int(epoch_from)) % epoch_every == 0
                         or (dense_from is not None and state["epoch"] >= int(dense_from)))):
                write_epoch_snapshot(state["epoch"])  # first optimizer step of a snapshot epoch
                last_epoch_saved = state["epoch"]
            max_hours = float(cfg["training"].get("max_hours", 0) or 0)  # wall-clock budget guard
            if max_hours and time.perf_counter() - start_time >= max_hours * 3600:
                save_checkpoint(latest, model, optimizer, scaler, state, cfg)
                save_checkpoint(output / f"step_{state['global_step']:07d}.pt", model, optimizer, scaler, state, cfg)
                if commit_callback is not None: commit_callback()
                print(json.dumps({"stopped_early": True, "reason": f"max_hours={max_hours} reached",
                                  "global_step": state["global_step"], "total_steps": total_steps}), flush=True)
                return {"checkpoint": str(latest), "state": state, "stopped_early": True}
            if state["global_step"] >= total_steps:
                if epoch_from is not None and cfg["training"].get("epoch_checkpoint_final", False):
                    # End of training: snapshot tagged with the number of completed epochs (e.g. epoch_150).
                    write_epoch_snapshot(state["epoch"] + (1 if state["batch_in_epoch"] >= len(loader) else 0),
                                         refresh_latest=not checkpoint_step)
                break
        if state["batch_in_epoch"] >= len(loader):
            state["epoch"] += 1; state["batch_in_epoch"] = 0
    return {"checkpoint": str(latest), "state": state}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True)
    args = parser.parse_args(); train(args.config)


if __name__ == "__main__": main()
