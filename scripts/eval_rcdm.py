"""Evaluate the trained RCDM visualizer on ImageNet-100 validation features (local GPU).

The I-JEPA paper uses RCDM to show what a representation keeps: qualities common across samples are
contained in the representation, qualities that vary across samples are not (Assran et al. 2023, Appendix E).
This script measures that on a class-stratified subset of the 5,000 validation images. For each image
it draws S samples from the image's pooled target-encoder feature, re-encodes every sample with the
same frozen encoder (`rcdm.pooled_target_features`, as the cache did), and reports:

  fidelity     cosine(sample feature, conditioning feature), against the VAE-reconstruction ceiling
               and real-image baselines (another validation image of the same class / of another class);
  retrieval    rank of the source image among all 5,000 validation features (top-1/5/10, median rank);
  class        weighted 20-NN (k=20, T=0.07, bank = the 126,689 train-cache features) accuracy of the
               samples vs the real images, and how often a sample gets the real image's predicted class;
  consistency  feature agreement between samples of one image vs their pixel-level variation.

Grids: one image per class for a spread of classes, plus the best and worst images by fidelity.
Sampling follows the training-time grids: EMA UNet, DDIM with eta = 0, fp32 with TF32 matmuls.

    python scripts/eval_rcdm.py --rcdm artifacts/rcdm_vitl16/step_0100000.pt \
        --encoder artifacts/vitl_snapshots/epoch_150_weights.pt --output-dir artifacts/rcdm_vitl16/evaluation
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

import src.ijepa  # noqa: F401  (imports `datasets` before torch on Windows)
import torch
import torch.nn.functional as F

from src.ijepa.data import IMAGENET_MEAN, IMAGENET_STD
from src.ijepa.evaluate import weighted_knn
from src.ijepa.rcdm import decode_latents, load_vae, pooled_target_features, sample
from src.ijepa.train_decoder import load_frozen_ijepa
from src.ijepa.train_rcdm import RCDMState, load_cache

VAL_PARQUET_NAME = "validation-00000-of-00001.parquet"  # inside --data-root (clane9/imagenet-100)
THUMB = 112


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def class_names(parquet: str, count: int) -> list[str]:
    """Short class names from the HF parquet metadata; indices if unavailable (names are cosmetic)."""
    try:
        import pyarrow.parquet as pq

        names = json.loads(pq.read_schema(parquet).metadata[b"huggingface"])["info"]["features"]["label"]["names"]
        return [name.split(",")[0] for name in names]
    except Exception:
        return [str(index) for index in range(count)]


@torch.inference_mode()
def knn_predict(bank: torch.Tensor, bank_labels: torch.Tensor, queries: torch.Tensor, k: int = 20,
                temperature: float = 0.07, batch_size: int = 256) -> torch.Tensor:
    """Predicted labels of `src.ijepa.evaluate.weighted_knn` (same normalisation, k, temperature, votes)."""
    classes = int(bank_labels.max()) + 1
    bank = F.normalize(bank.float(), dim=-1)
    queries = F.normalize(queries.float(), dim=-1)
    predictions = []
    for start in range(0, len(queries), batch_size):
        similarity = queries[start:start + batch_size] @ bank.T
        values, indices = similarity.topk(min(k, len(bank)), dim=1)
        votes = torch.zeros(similarity.size(0), classes)
        votes.scatter_add_(1, bank_labels[indices], (values / temperature).exp())
        predictions.append(votes.argmax(dim=1))
    return torch.cat(predictions)


def source_ranks(queries: torch.Tensor, gallery: torch.Tensor, sources: torch.Tensor) -> torch.Tensor:
    """1-based rank of gallery[sources[i]] among all gallery items by cosine to queries[i]."""
    similarity = F.normalize(queries.float(), dim=-1) @ F.normalize(gallery.float(), dim=-1).T
    own = similarity.gather(1, sources.view(-1, 1))
    return (similarity > own).sum(dim=1) + 1


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.float(), b.float(), dim=-1)


def stratified(labels: torch.Tensor, per_class: int, classes: int) -> torch.Tensor:
    picks = [torch.nonzero(labels == c).flatten()[:per_class] for c in range(classes)]
    return torch.cat(picks)


def other_index(labels: torch.Tensor, index: int, same_class: bool, generator: torch.Generator) -> int:
    pool = torch.nonzero((labels == labels[index]) if same_class else (labels != labels[index])).flatten()
    pool = pool[pool != index]
    return int(pool[torch.randint(len(pool), (1,), generator=generator)])


@torch.inference_mode()
def decode(vae, latents: torch.Tensor, chunk: int = 10) -> torch.Tensor:
    return torch.cat([decode_latents(vae, latents[i:i + chunk]) for i in range(0, len(latents), chunk)])


@torch.inference_mode()
def encode(encoder, images01: torch.Tensor, chunk: int = 50) -> torch.Tensor:
    mean = images01.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = images01.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(images01), chunk):
        with torch.autocast("cuda", torch.bfloat16, enabled=images01.is_cuda):  # as in rcdm_cache.build_cache
            out.append(pooled_target_features(encoder, (images01[i:i + chunk] - mean) / std).float().cpu())
    return torch.cat(out)


def to_uint8(images01: torch.Tensor, size: int | None = None) -> torch.Tensor:
    if size is not None and images01.size(-1) != size:
        images01 = F.interpolate(images01, size=(size, size), mode="area")
    return (images01.clamp(0, 1) * 255).round().to(torch.uint8).cpu()


def save_grid(path: Path, rows: list[torch.Tensor], captions: list[str]) -> None:
    """rows: uint8 [1+S,3,H,W] each (reconstruction then samples); caption drawn on the first tile."""
    from PIL import Image, ImageDraw

    tiles, height = rows[0].size(0), rows[0].size(-1)
    pad = 2
    canvas = Image.new("RGB", (tiles * (height + pad) + pad, len(rows) * (height + pad) + pad), "white")
    draw = ImageDraw.Draw(canvas)
    for r, (row, caption) in enumerate(zip(rows, captions)):
        for t in range(tiles):
            x, y = pad + t * (height + pad), pad + r * (height + pad)
            canvas.paste(Image.fromarray(row[t].permute(1, 2, 0).numpy()), (x, y))
        box = draw.textbbox((pad + 2, pad + r * (height + pad) + 2), caption)
        draw.rectangle(box, fill="black")
        draw.text((box[0], box[1]), caption, fill="white")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def summarize(values: torch.Tensor) -> dict:
    values = values.float().flatten()
    return {"mean": round(float(values.mean()), 4), "std": round(float(values.std()), 4),
            "median": round(float(values.median()), 4)}


def retrieval_summary(ranks: torch.Tensor, gallery: int) -> dict:
    ranks = ranks.flatten()
    return {f"top{k}": round(float((ranks <= k).float().mean()), 4) for k in (1, 5, 10)} | {
        "median_rank": float(ranks.float().median()), "gallery": gallery}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rcdm", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--val-cache", default="artifacts/rcdm_cache/imagenet100_vitl16_validation.pt")
    parser.add_argument("--train-cache", default="artifacts/rcdm_cache/imagenet100_vitl16_train.pt")
    parser.add_argument("--vae-cache", default=None, help="Hugging Face cache dir for sd-vae-ft-mse (default: HF default)")
    parser.add_argument("--data-root", default="data/imagenet-100/data", help="ImageNet-100 parquet folder (class names)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--classes", type=int, default=100, help="use only the first N classes (quick checks)")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--grid-classes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    started = time.time()
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True  # as in train_rcdm (its grids sample in this mode)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    val = load_cache(args.val_cache)
    train = load_cache(args.train_cache)
    bank, bank_labels = train["features"], train["labels"]
    del train
    labels, features = val["labels"], val["features"]
    names = class_names(str(Path(args.data_root) / VAL_PARQUET_NAME), int(labels.max()) + 1)
    print(json.dumps({"val": len(labels), "bank": len(bank_labels)}), flush=True)

    # Sanity: the cached features must reproduce the published k-NN protocol (60.02% from images).
    cache_knn_full = weighted_knn(bank, bank_labels, features, labels)
    real_pred_full = knn_predict(bank, bank_labels, features)
    assert int((real_pred_full == labels).sum()) == round(cache_knn_full * len(labels))  # same predictions

    selected = stratified(labels, args.per_class, args.classes)
    grid_every = max(1, args.classes // args.grid_classes)
    grid_ids = {int(selected[(labels[selected] == c).nonzero()[0, 0]]) for c in range(0, args.classes, grid_every)}

    checkpoint = torch.load(args.rcdm, map_location="cpu", weights_only=False)
    state = RCDMState(checkpoint["config"]["model"])
    state.load_state_dict(checkpoint["model"])
    ema = state.ema.to(device).eval()
    ijepa, encoder_ckpt = load_frozen_ijepa(args.encoder, torch.device("cpu"))
    encoder = SimpleNamespace(target_encoder=ijepa.target_encoder.to(device))
    del ijepa
    vae = load_vae(device, cache_dir=args.vae_cache)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    recon_feats, sample_feats = [], []
    pixel_within, pixel_to_recon, pixel_unrelated = [], [], []
    thumbs, full_rows = {}, {}
    sampling_start = time.time()
    for start in range(0, len(selected), args.batch_size):
        batch = selected[start:start + args.batch_size]
        cond = features[batch].float().to(device)
        recon = decode(vae, val["latents"][batch].float().to(device))
        draws = []
        for _ in range(args.samples):
            draws.append(decode(vae, sample(ema, cond, steps=args.steps, generator=generator)))
        draws = torch.stack(draws, 1)  # [B,S,3,224,224] in [0,1]
        recon_feats.append(encode(encoder, recon))
        sample_feats.append(encode(encoder, draws.flatten(0, 1)).view(len(batch), args.samples, -1))
        pairs = [(a, b) for a in range(args.samples) for b in range(a + 1, args.samples)]
        if pairs:
            pixel_within.append(torch.stack([(draws[:, a] - draws[:, b]).abs().mean((1, 2, 3)) for a, b in pairs], 1).cpu())
        pixel_to_recon.append((draws - recon.unsqueeze(1)).abs().mean((2, 3, 4)).cpu())
        shift = args.per_class if len(batch) > args.per_class else 1  # batches are class-ordered: next class
        pixel_unrelated.append((recon - recon.roll(shift, 0)).abs().mean((1, 2, 3)).cpu())
        for row, index in enumerate(batch.tolist()):
            tiles = torch.cat([recon[row:row + 1], draws[row]])
            thumbs[index] = to_uint8(tiles, THUMB)
            if index in grid_ids:
                full_rows[index] = to_uint8(tiles)
        done = start + len(batch)
        rate = done * args.samples / (time.time() - sampling_start)
        print(json.dumps({"images": done, "of": len(selected), "samples_per_s": round(rate, 2)}), flush=True)

    recon_feats = torch.cat(recon_feats)              # [N,D]
    sample_feats = torch.cat(sample_feats)            # [N,S,D]
    cond = features[selected].float()                 # [N,D]
    n, s = sample_feats.shape[:2]

    # Fidelity: raw cosine (the k-NN space) and cosine after removing the train-feature mean.
    center = bank.float().mean(0)
    rng = torch.Generator().manual_seed(args.seed)
    same = torch.tensor([other_index(labels, int(i), True, rng) for i in selected])
    other = torch.tensor([other_index(labels, int(i), False, rng) for i in selected])
    fidelity = {}
    for space, shift in (("raw", 0.0), ("centered", center)):
        c = cond - shift
        fidelity[space] = {
            "sample_vs_its_feature": summarize(cosine(sample_feats - shift, c.unsqueeze(1))),
            "vae_reconstruction_vs_its_feature": summarize(cosine(recon_feats - shift, c)),
            "other_image_same_class": summarize(cosine(features[same].float() - shift, c)),
            "other_image_other_class": summarize(cosine(features[other].float() - shift, c)),
        }
    per_image_fidelity = cosine(sample_feats, cond.unsqueeze(1)).mean(1)  # raw, [N]

    # Retrieval of the source among all validation features.
    sample_ranks = source_ranks(sample_feats.flatten(0, 1), features, selected.repeat_interleave(s)).view(n, s)
    recon_ranks = source_ranks(recon_feats, features, selected)

    # Class preservation with the k-NN protocol.
    true = labels[selected]
    real_pred = real_pred_full[selected]
    recon_pred = knn_predict(bank, bank_labels, recon_feats)
    sample_pred = knn_predict(bank, bank_labels, sample_feats.flatten(0, 1)).view(n, s)
    per_class = {}
    for c in range(args.classes):
        rows = true == c
        per_class[f"{c}:{names[c]}"] = round(float((sample_pred[rows] == c).float().mean()), 3)
    ranked_classes = sorted(per_class.items(), key=lambda kv: kv[1])

    # Consistency: samples of one image vs samples of different images.
    pair_cos = [cosine(sample_feats[:, a], sample_feats[:, b]) for a in range(s) for b in range(a + 1, s)]
    same_class_pairs, other_class_pairs = [], []
    position = {int(index): row for row, index in enumerate(selected)}
    for row, index in enumerate(selected.tolist()):
        mates = [position[int(j)] for j in selected[(true == true[row])].tolist() if int(j) != index]
        same_class_pairs.append(cosine(sample_feats[row, 0], sample_feats[mates[0], 0]) if mates else torch.tensor(float("nan")))
        other_class_pairs.append(cosine(sample_feats[row, 0], sample_feats[(row + args.per_class) % n, 0]))
    same_class_pairs = torch.stack(same_class_pairs)
    consistency = {
        "feature_cosine_between_samples_of_one_image": summarize(torch.stack(pair_cos, 1)) if pair_cos else None,
        "feature_cosine_samples_of_two_images_same_class": summarize(same_class_pairs[~same_class_pairs.isnan()]),
        "feature_cosine_samples_of_two_images_other_class": summarize(torch.stack(other_class_pairs)),
        "pixel_l1_between_samples_of_one_image": summarize(torch.cat(pixel_within)) if pixel_within else None,
        "pixel_l1_sample_vs_vae_reconstruction": summarize(torch.cat(pixel_to_recon)),
        "pixel_l1_reconstructions_of_two_other_class_images": summarize(torch.cat(pixel_unrelated)),
    }

    results = {
        "protocol": {
            "rcdm_checkpoint": args.rcdm, "rcdm_sha256": sha256(args.rcdm),
            "rcdm_step": int(checkpoint["state"]["global_step"]), "weights": "EMA UNet",
            "encoder": args.encoder, "encoder_step": int(encoder_ckpt["state"]["global_step"]),
            "conditioning": "average-pooled target-encoder feature of the validation image (cached)",
            "images": n, "per_class": args.per_class, "classes": args.classes, "samples_per_image": s,
            "ddim_steps": args.steps, "eta": 0.0, "precision": "fp32 + TF32 matmuls (UNet, VAE); bf16 autocast (encoder)",
            "seed": args.seed, "knn": "weighted, k=20, T=0.07, bank = 126,689 train-cache features",
        },
        "sanity_knn_of_cached_val_features_all_5000": round(cache_knn_full, 4),
        "fidelity_cosine": fidelity,
        "source_retrieval": {"samples": retrieval_summary(sample_ranks, len(features)),
                             "vae_reconstruction": retrieval_summary(recon_ranks, len(features))},
        "class_knn_accuracy": {
            "real_images": round(float((real_pred == true).float().mean()), 4),
            "vae_reconstructions": round(float((recon_pred == true).float().mean()), 4),
            "samples": round(float((sample_pred == true.unsqueeze(1)).float().mean()), 4),
            "samples_agree_with_real_image_prediction": round(float((sample_pred == real_pred.unsqueeze(1)).float().mean()), 4),
            "chance": round(1 / (int(bank_labels.max()) + 1), 4),  # the vote is always over all classes
        },
        "class_sample_accuracy_worst5": ranked_classes[:5],
        "class_sample_accuracy_best5": ranked_classes[-5:][::-1],
        "consistency": consistency,
        "seconds": round(time.time() - started, 1),
    }

    order = torch.argsort(per_image_fidelity)
    per_image = []
    for row in range(n):
        index = int(selected[row])
        per_image.append({"val_index": index, "label": int(true[row]), "class": names[int(true[row])],
                          "fidelity_cosine": round(float(per_image_fidelity[row]), 4),
                          "sample_ranks": sample_ranks[row].tolist(), "recon_rank": int(recon_ranks[row]),
                          "real_pred": names[int(real_pred[row])],
                          "sample_preds": [names[int(p)] for p in sample_pred[row]]})

    def caption(row: int) -> str:
        return (f"{names[int(true[row])][:18]} cos {float(per_image_fidelity[row]):.2f} "
                f"rank {int(sample_ranks[row].median())}")

    grid_rows = [row for row in range(n) if int(selected[row]) in full_rows]
    half = (len(grid_rows) + 1) // 2
    grids = []
    for part, rows in enumerate((grid_rows[:half], grid_rows[half:]), start=1):
        if rows:
            path = output / f"classes_{part}.png"
            save_grid(path, [full_rows[int(selected[r])] for r in rows], [caption(r) for r in rows])
            grids.append(str(path))
    for name, rows in (("worst", order[:12].tolist()), ("best", order[-12:].flip(0).tolist())):
        path = output / f"{name}_fidelity.png"
        save_grid(path, [thumbs[int(selected[r])] for r in rows], [caption(r) for r in rows])
        grids.append(str(path))
    results["grids"] = grids + ["each row: VAE reconstruction of the real image, then the samples; caption: class, "
                                "mean sample cosine to the conditioning feature, median source rank among the 5,000 validation features"]

    (output / "rcdm_evaluation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output / "per_image.json").write_text(json.dumps(per_image, indent=1), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
