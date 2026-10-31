"""Non-generative visualizations of an I-JEPA encoder on STL-10.

Writes to ``--output-dir``:
  mask_overlay.png   -- training-style masks: context patches green, target blocks red
  pca_patches.png    -- row 1 originals, row 2 PCA(3) of target-encoder patch tokens as RGB
  knn_retrieval.png  -- each row: a test query (first column) and its top-k train neighbours
  summary.json       -- neighbour label agreement and the images/classes used

Example:
  python scripts/visualize_encoder.py --checkpoint artifacts/modal_encoder_step_0200000_final.pt \
      --output-dir artifacts/visualization/encoder_step200k
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `src` imports work

import src.ijepa  # noqa: F401  (imports `datasets` before torch; see src/ijepa/__init__.py)
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid

from src.ijepa.data import make_dataset
from src.ijepa.evaluate import extract_features
from src.ijepa.masks import MultiBlockMaskSampler
from src.ijepa.train_decoder import denormalize, load_frozen_ijepa


def one_per_class(dataset, count: int) -> list[int]:
    """First test index of each of the first `count` classes (deterministic)."""
    chosen, seen = [], set()
    labels = dataset.labels
    for index, label in enumerate(labels):
        if int(label) not in seen:
            seen.add(int(label))
            chosen.append(index)
        if len(chosen) == count:
            break
    return chosen


def save_rows(path: Path, images: torch.Tensor, ncol: int) -> None:
    grid = make_grid(images.clamp(0, 1), nrow=ncol, padding=2, pad_value=1.0)
    Image.fromarray((grid.permute(1, 2, 0).numpy() * 255).round().astype("uint8")).save(path)


def mask_overlay(images: torch.Tensor, masks, patch: int) -> torch.Tensor:
    b, _, size, _ = images.shape
    grid = size // patch
    base = denormalize(images).cpu()
    tint = torch.zeros(b, 3, grid * grid)
    alpha = torch.zeros(b, 1, grid * grid)
    for i in range(b):
        tint[i, 1, masks.context[i].cpu()] = 1.0
        alpha[i, 0, masks.context[i].cpu()] = 0.45
        target = masks.targets[:, i].flatten().cpu()
        tint[i, :, target] = torch.tensor([1.0, 0.0, 0.0]).view(3, 1)
        alpha[i, 0, target] = 0.55
    up = lambda x: F.interpolate(x.view(b, -1, grid, grid), scale_factor=patch, mode="nearest")
    tint, alpha = up(tint), up(alpha)
    return base * (1 - alpha) + tint * alpha


@torch.inference_mode()
def pca_rgb(tokens: torch.Tensor, grid: int, patch: int) -> torch.Tensor:
    """Joint PCA over all images' patch tokens -> [N,3,H,W] in [0,1]."""
    n, t, d = tokens.shape
    flat = tokens.reshape(-1, d).float()
    flat = flat - flat.mean(0, keepdim=True)
    _u, _s, v = torch.pca_lowrank(flat, q=3, center=False)
    proj = flat @ v[:, :3]
    low, high = proj.quantile(0.01, dim=0), proj.quantile(0.99, dim=0)
    proj = ((proj - low) / (high - low)).clamp(0, 1)
    rgb = proj.reshape(n, grid, grid, 3).permute(0, 3, 1, 2)
    return F.interpolate(rgb, scale_factor=patch, mode="nearest")


def label_strip(image: Image.Image, labels: list[str], cell: int, pad: int = 2) -> Image.Image:
    strip = Image.new("RGB", (image.width, 16), "white")
    draw = ImageDraw.Draw(strip)
    for j, text in enumerate(labels):
        draw.text((pad + j * (cell + pad) + 2, 2), text[:18], fill="black")
    out = Image.new("RGB", (image.width, image.height + 16), "white")
    out.paste(image, (0, 0))
    out.paste(strip, (0, image.height))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--images", type=int, default=8)
    parser.add_argument("--neighbours", type=int, default=5)
    parser.add_argument("--mask-seed", type=int, default=20260831)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_frozen_ijepa(args.checkpoint, device)
    cfg = checkpoint["config"]
    size, patch = int(cfg["model"]["image_size"]), int(cfg["model"]["patch_size"])
    grid = size // patch
    data_cfg = {**cfg["data"], "root": args.data_root}
    test_set = make_dataset(data_cfg, train=False, split="test")
    train_set = make_dataset(data_cfg, train=False, split="train")
    classes = (Path(args.data_root) / "stl10_binary" / "class_names.txt").read_text().split()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    query_ids = one_per_class(test_set, args.images)
    queries = torch.stack([test_set[i][0] for i in query_ids]).to(device)

    # 1. Masks exactly as training samples them (batch-minimum context trimming included).
    sampler = MultiBlockMaskSampler(**{**cfg["mask"], "image_size": size, "patch_size": patch})
    masks = sampler.sample(queries.size(0), torch.Generator().manual_seed(args.mask_seed))
    save_rows(out / "mask_overlay.png", torch.cat([denormalize(queries).cpu(), mask_overlay(queries, masks, patch)]), queries.size(0))

    # 2. PCA of target-encoder patch tokens (the representation used for evaluation).
    with torch.inference_mode(), torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
        tokens = model.target_encoder(queries)
    rgb = pca_rgb(tokens.float().cpu(), grid, patch)
    save_rows(out / "pca_patches.png", torch.cat([denormalize(queries).cpu(), rgb]), queries.size(0))

    # 3. k-NN retrieval with the same features as the k-NN evaluation.
    loader = lambda ds: DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
    bank, bank_labels = extract_features(model.target_encoder, loader(train_set), device)
    query_features, query_labels = extract_features(model.target_encoder, loader(Subset(test_set, query_ids)), device)
    similarity = query_features @ bank.T
    top = similarity.topk(args.neighbours, dim=1).indices
    rows, agreement, names = [], [], []
    for q, neighbour_ids in enumerate(top):
        rows.append(denormalize(queries[q:q + 1]).cpu())
        row_names = [f"Q:{classes[int(query_labels[q])]}"]
        for idx in neighbour_ids.tolist():
            rows.append(denormalize(train_set[idx][0].unsqueeze(0)))
            row_names.append(classes[int(bank_labels[idx])])
            agreement.append(int(bank_labels[idx]) == int(query_labels[q]))
        names.append(row_names)
    save_rows(out / "knn_retrieval.png", torch.cat(rows), args.neighbours + 1)
    grid_image = Image.open(out / "knn_retrieval.png").convert("RGB")
    cell = size
    row_h = cell + 2
    annotated = Image.new("RGB", (grid_image.width, 0), "white")
    for r, row_names in enumerate(names):
        row = grid_image.crop((0, r * row_h, grid_image.width, (r + 1) * row_h + 2))
        row = label_strip(row, row_names, cell)
        merged = Image.new("RGB", (grid_image.width, annotated.height + row.height), "white")
        merged.paste(annotated, (0, 0))
        merged.paste(row, (0, annotated.height))
        annotated = merged
    annotated.save(out / "knn_retrieval.png")

    summary = {
        "checkpoint": args.checkpoint,
        "encoder_step": int(checkpoint["state"]["global_step"]),
        "query_test_indices": query_ids,
        "query_classes": [classes[int(l)] for l in query_labels],
        "neighbours": args.neighbours,
        "neighbour_label_agreement": sum(agreement) / len(agreement),
        "per_query_neighbour_classes": names,
        "mask_seed": args.mask_seed,
        "context_tokens_per_image": int(masks.context.shape[1]),
        "target_tokens_per_block": int(masks.targets.shape[2]),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("encoder_step", "neighbour_label_agreement", "query_classes")}))


if __name__ == "__main__":
    main()
