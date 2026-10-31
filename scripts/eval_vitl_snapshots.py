"""Weighted 20-NN evaluation of ViT-L I-JEPA weight snapshots on ImageNet-100 (local GPU).

Bank = ImageNet-100 train (126,689, eval transform); queries = validation (5,000). Uses the
target encoder, average-pooled patch tokens, cosine similarity, k=20, T=0.07 (the same
protocol as the STL-10 k-NN numbers). `--random-init` also evaluates an untrained ViT-L with
the same architecture as the missing baseline.

    python scripts/eval_vitl_snapshots.py artifacts/vitl_snapshots/epoch_075_weights.pt [...] --random-init \
        [--data-root data/imagenet-100/data] [--cache-dir data/hf_cache]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

import src.ijepa  # noqa: F401  (imports `datasets` before torch on Windows)
import torch
from torch.utils.data import DataLoader

from src.ijepa.data import make_dataset
from src.ijepa.evaluate import extract_features, weighted_knn
from src.ijepa.model import IJEPAModel

DEFAULT_DATA_ROOT = "data/imagenet-100/data"  # clane9/imagenet-100 parquet shards
DEFAULT_CACHE_DIR = "data/hf_cache"


def load_encoder(path: str | None, reference_cfg: dict, device: torch.device):
    model = IJEPAModel(**reference_cfg["model"])
    step = epoch = None
    if path is not None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])  # bf16 snapshot -> fp32 params (copy_ casts)
        step, epoch = payload["state"]["global_step"], payload["state"]["epoch"]
    return model.target_encoder.to(device).eval(), step, epoch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshots", nargs="+")
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--output", default="artifacts/vitl_snapshots/knn_imagenet100.json")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="folder holding the ImageNet-100 parquet shards")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help="Hugging Face datasets cache")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref = torch.load(args.snapshots[0], map_location="cpu", weights_only=False)["config"]
    data_cfg = {**ref["data"], "root": args.data_root, "cache_dir": args.cache_dir}
    bank_set = make_dataset(data_cfg, train=False, split="train")
    query_set = make_dataset(data_cfg, train=False, split="test")   # -> validation split
    loader = lambda ds: DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.num_workers, pin_memory=True)
    runs = [(p, p) for p in args.snapshots] + ([("random_init", None)] if args.random_init else [])
    results = {}
    for name, path in runs:
        start = time.perf_counter()
        encoder, step, epoch = load_encoder(path, ref, device)
        with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
            bank, bank_labels = extract_features(encoder, loader(bank_set), device)
            query, query_labels = extract_features(encoder, loader(query_set), device)
        accuracy = weighted_knn(bank, bank_labels, query, query_labels)
        results[str(name)] = {"checkpoint": path, "global_step": step, "epoch": epoch,
                              "imagenet100_val_knn20_accuracy": accuracy,
                              "bank": len(bank_labels), "queries": len(query_labels),
                              "seconds": round(time.perf_counter() - start, 1)}
        print(json.dumps({k: results[str(name)][k] for k in ("epoch", "imagenet100_val_knn20_accuracy", "seconds")} | {"name": str(name)}), flush=True)
        del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"protocol": {"dataset": "ImageNet-100 (clane9, 160 px short side)", "bank": "train, eval transform",
                                            "queries": "validation", "k": 20, "temperature": 0.07,
                                            "features": "target encoder, average-pooled, L2-normalized"},
                               "results": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
