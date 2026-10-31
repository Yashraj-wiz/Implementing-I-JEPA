from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timedelta
from pathlib import Path

import yaml


def duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}h {minutes:02d}m {seconds:02d}s"


def read_last_metric(path: Path) -> dict:
    last = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last = json.loads(line)
    if last is None:
        raise RuntimeError(f"No metrics have been written to {path}")
    return last


def render(config_path: Path) -> None:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    training = config["training"]
    batch_size = int(config["data"]["batch_size"])
    total_steps = int(training["total_steps"])
    output_dir = Path(training["output_dir"])
    metrics_path = output_dir / "metrics.jsonl"
    checkpoint_path = output_dir / "latest.pt"
    metric = read_last_metric(metrics_path)

    step = int(metric["step"])
    throughput = float(metric["images_per_second"])
    completed_images = step * batch_size
    total_images = total_steps * batch_size
    elapsed = completed_images / throughput
    remaining = max(0, total_images - completed_images) / throughput
    checkpoint_every = int(training.get("checkpoint_every", total_steps))
    checkpoint_step = math.floor(step / checkpoint_every) * checkpoint_every
    eta = datetime.now().astimezone() + timedelta(seconds=remaining)

    print(f"step             {step:,} / {total_steps:,} ({step / total_steps:.2%})")
    print(f"images           {completed_images:,} / {total_images:,}")
    print(f"loss             {float(metric['loss']):.6f}")
    print(f"throughput       {throughput:.1f} images/s")
    print(f"estimated time   {duration(elapsed)} elapsed, {duration(remaining)} remaining")
    print(f"estimated finish {eta:%Y-%m-%d %H:%M:%S %Z}")
    print(f"peak GPU memory  {float(metric['peak_memory_gb']):.2f} GiB")
    print(f"feature std      {float(metric['feature_std']):.4f}")
    print(f"mean cosine      {float(metric['mean_cosine']):.4f}")
    if checkpoint_path.exists():
        updated = datetime.fromtimestamp(checkpoint_path.stat().st_mtime).astimezone()
        print(
            f"last checkpoint  step {checkpoint_step:,}, "
            f"{checkpoint_path} ({updated:%Y-%m-%d %H:%M:%S %Z})"
        )
    else:
        print("last checkpoint  none yet")


def main() -> None:
    parser = argparse.ArgumentParser(description="Show I-JEPA training progress")
    parser.add_argument(
        "--config",
        default="configs/stl10_vits16_local.yaml",
        type=Path,
    )
    parser.add_argument(
        "--watch",
        type=float,
        metavar="SECONDS",
        help="Refresh continuously at this interval",
    )
    args = parser.parse_args()

    while True:
        render(args.config)
        if args.watch is None:
            return
        print()
        time.sleep(max(1.0, args.watch))


if __name__ == "__main__":
    main()
