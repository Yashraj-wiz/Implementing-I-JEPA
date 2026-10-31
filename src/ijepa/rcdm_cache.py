"""Precompute (VAE latent, pooled I-JEPA feature) pairs once, so RCDM training never runs the
VAE encoder or the I-JEPA model per step (decision D-011).

Both are computed from the same deterministic eval-transform crop (Resize -> CenterCrop 224),
so each feature describes exactly the pixels its latent encodes.

    python -m src.ijepa.rcdm_cache --encoder <ijepa.pt> --split train --output <cache.pt> \
        [--data-root data] [--limit N] [--vae-cache DIR]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from .data import make_dataset
from .rcdm import encode_latents, load_vae, pooled_target_features
from .train_decoder import load_frozen_ijepa


@torch.inference_mode()
def build_cache(encoder_path: str, split: str, output: str, data_root: str | None = None,
                limit: int | None = None, batch_size: int = 64, num_workers: int = 0,
                vae_cache: str | None = None, device: str | None = None) -> dict:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, checkpoint = load_frozen_ijepa(encoder_path, dev)
    data_cfg = dict(checkpoint["config"]["data"])
    if data_root is not None:
        data_cfg["root"] = data_root
    dataset = make_dataset(data_cfg, train=False, split=split)  # deterministic eval transform
    if limit is not None:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    vae = load_vae(dev, cache_dir=vae_cache)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=dev.type == "cuda")
    latents, features, labels = [], [], []
    start = time.perf_counter()
    for images, target in loader:
        images = images.to(dev, non_blocking=True)
        with torch.autocast(dev.type, torch.bfloat16, enabled=dev.type == "cuda"):
            feats = pooled_target_features(model, images)
        latents.append(encode_latents(vae, images).half().cpu())
        features.append(feats.float().half().cpu())
        labels.append(torch.as_tensor(target).long())
    payload = {
        "latents": torch.cat(latents),      # [N,4,28,28] fp16, scaled by vae.config.scaling_factor
        "features": torch.cat(features),    # [N,D] fp16, average-pooled target-encoder output
        "labels": torch.cat(labels),
        "meta": {
            "encoder_checkpoint": encoder_path,
            "encoder_step": int(checkpoint["state"]["global_step"]),
            "encoder_model": checkpoint["config"]["model"],
            "dataset": data_cfg["name"], "split": split, "count": len(dataset),
            "transform": "eval (Resize image_size+32 -> CenterCrop image_size)",
            "vae": "stabilityai/sd-vae-ft-mse (posterior mean x scaling_factor)",
        },
    }
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(out)
    summary = {**payload["meta"], "seconds": round(time.perf_counter() - start, 1),
               "latent_std": float(payload["latents"].float().std()),
               "feature_dim": int(payload["features"].shape[1]), "output": str(out)}
    print(json.dumps(summary), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--split", required=True, help="dataset split for eval transforms, e.g. train or test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--vae-cache")
    parser.add_argument("--device")
    args = parser.parse_args()
    build_cache(args.encoder, args.split, args.output, args.data_root, args.limit, args.batch_size,
                args.num_workers, args.vae_cache, args.device)


if __name__ == "__main__":
    main()
