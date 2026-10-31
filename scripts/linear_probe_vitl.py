"""Frozen-encoder linear probe on ImageNet-100 for ViT-L I-JEPA snapshots (local GPU).

Follows the paper's probe recipe (Assran et al. 2023, Appendix A.2) on cached features:
- representation: average-pooled last-layer target-encoder tokens ("last"), or the concatenation of
  the average-pooled outputs of the last 4 blocks ("last4"), each passed through the encoder's final LayerNorm;
- head: BatchNorm1d(affine=False) + Linear, as in MAE's linear probe;
- LARS, batch 16,384, 50 epochs, LR divided by 10 every 15 epochs, reference LRs {0.01, 0.05, 0.001}
  scaled by batch/256, weight decay {0.0005, 0.0}.

Deviation: features are extracted once with the eval transform (Resize 256 -> CenterCrop 224), so the probe
trains without random-crop/flip augmentation.

Model selection is reported two ways:
- "selected": hyperparameters chosen on a held-out 10% of train, then retrained on the full train split
  and scored once on validation (the honest number);
- "best_on_val": the maximum validation accuracy over the sweep (how the paper reports it).

    python scripts/linear_probe_vitl.py artifacts/vitl_snapshots/epoch_150_weights.pt --random-init
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

import src.ijepa  # noqa: F401  (imports `datasets` before torch on Windows)
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.ijepa.data import make_dataset
from src.ijepa.model import IJEPAModel

DEFAULT_DATA_ROOT = "data/imagenet-100/data"  # clane9/imagenet-100 parquet shards
DEFAULT_CACHE_DIR = "data/hf_cache"
REF_LRS = (0.01, 0.05, 0.001)
WEIGHT_DECAYS = (0.0005, 0.0)


@torch.inference_mode()
def extract(encoder, loader, device):
    """Return (last [N,D], last4 [N,4D], labels [N]) as fp16/int64 on CPU."""
    lasts, last4s, labels = [], [], []
    for images, target in loader:
        x = encoder.patch_embed(images.to(device, non_blocking=True)).flatten(2).transpose(1, 2)
        x = x + encoder.pos_embed.to(dtype=x.dtype)
        outs = []
        for i, block in enumerate(encoder.blocks):
            x = block(x)
            if i >= len(encoder.blocks) - 4:
                outs.append(encoder.norm(x).mean(dim=1))
        lasts.append(outs[-1].half().cpu())
        last4s.append(torch.cat(outs, dim=1).half().cpu())
        labels.append(target.long())
    return torch.cat(lasts), torch.cat(last4s), torch.cat(labels)


class LARS(torch.optim.Optimizer):
    """LARS as in MAE's util/lars.py: weight decay and trust ratio only for >1-D parameters."""

    def __init__(self, params, lr, weight_decay=0.0, momentum=0.9, trust_coefficient=0.001):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                                      trust_coefficient=trust_coefficient))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                dp = p.grad
                if p.ndim > 1:
                    dp = dp.add(p, alpha=group["weight_decay"])
                    param_norm, update_norm = torch.norm(p), torch.norm(dp)
                    q = torch.where((param_norm > 0) & (update_norm > 0),
                                    group["trust_coefficient"] * param_norm / update_norm, torch.ones_like(param_norm))
                    dp = dp.mul(q)
                buf = self.state[p].setdefault("mu", torch.zeros_like(p))
                buf.mul_(group["momentum"]).add_(dp)
                p.add_(buf, alpha=-group["lr"])


def train_probe(x, y, num_classes, ref_lr, wd, device, epochs=50, batch=16384, seed=0):
    torch.manual_seed(seed)
    head = nn.Sequential(nn.BatchNorm1d(x.size(1), affine=False, eps=1e-6), nn.Linear(x.size(1), num_classes)).to(device)
    nn.init.trunc_normal_(head[1].weight, std=0.01); nn.init.zeros_(head[1].bias)
    base_lr = ref_lr * batch / 256
    opt = LARS(head.parameters(), lr=base_lr, weight_decay=wd)
    n = x.size(0)
    for epoch in range(epochs):
        for g in opt.param_groups:
            g["lr"] = base_lr * (0.1 ** (epoch // 15))
        head.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            loss = nn.functional.cross_entropy(head(x[idx].float()), y[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if not torch.isfinite(loss):
            return head, float("nan")
    return head, float(loss)


@torch.inference_mode()
def accuracy(head, x, y, chunk=8192):
    head.eval()
    top1 = top5 = 0
    for s in range(0, x.size(0), chunk):
        logits = head(x[s:s + chunk].float())
        t5 = logits.topk(5, dim=1).indices
        top1 += (t5[:, 0] == y[s:s + chunk]).sum().item()
        top5 += (t5 == y[s:s + chunk, None]).any(dim=1).sum().item()
    return top1 / x.size(0), top5 / x.size(0)


def load_encoder(path, cfg, device):
    model = IJEPAModel(**cfg["model"])
    info = {"checkpoint": path, "global_step": None, "epoch": None}
    if path is not None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        info.update(global_step=payload["state"]["global_step"], epoch=payload["state"]["epoch"])
    return model.target_encoder.to(device).eval(), info


def sweep(features, device, num_classes, log, ref_lrs=REF_LRS):
    """features: {rep: (train_x, train_y, val_x, val_y)} on device."""
    runs = []
    for rep, (tx, ty, vx, vy) in features.items():
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(tx.size(0), generator=g).to(device)
        n_hold = tx.size(0) // 10
        hold, fit = perm[:n_hold], perm[n_hold:]
        for ref_lr, wd in itertools.product(ref_lrs, WEIGHT_DECAYS):
            head, loss = train_probe(tx[fit], ty[fit], num_classes, ref_lr, wd, device)
            h1, _ = accuracy(head, tx[hold], ty[hold])
            head, loss_full = train_probe(tx, ty, num_classes, ref_lr, wd, device)
            v1, v5 = accuracy(head, vx, vy)
            run = {"rep": rep, "ref_lr": ref_lr, "weight_decay": wd, "holdout_top1": h1,
                   "val_top1": v1, "val_top5": v5, "final_train_loss": loss_full}
            runs.append(run); log(json.dumps(run))
    chosen = max(runs, key=lambda r: r["holdout_top1"])
    best = max(runs, key=lambda r: r["val_top1"])
    return {"selected": chosen, "best_on_val": best, "runs": runs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshots", nargs="+")
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128, help="feature-extraction batch size")
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--output", default="artifacts/vitl_snapshots/linear_probe_imagenet100.json")
    parser.add_argument("--feature-cache-dir", default="artifacts/vitl_snapshots/probe_features")
    parser.add_argument("--ref-lrs", type=float, nargs="+", default=list(REF_LRS), help="reference LRs (scaled by batch/256)")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="folder holding the ImageNet-100 parquet shards")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help="Hugging Face datasets cache")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref = torch.load(args.snapshots[0], map_location="cpu", weights_only=False)["config"]
    data_cfg = {**ref["data"], "root": args.data_root, "cache_dir": args.cache_dir}
    train_set = make_dataset(data_cfg, train=False, split="train")
    val_set = make_dataset(data_cfg, train=False, split="test")  # -> validation split
    loader = lambda ds: DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    cache_dir = Path(args.feature_cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out.read_text(encoding="utf-8"))["results"] if out.exists() else {}
    log = lambda s: print(s, flush=True)

    for path in list(args.snapshots) + ([None] if args.random_init else []):
        name = "random_init" if path is None else Path(path).stem
        start = time.perf_counter()
        cache = cache_dir / f"{name}.pt"
        if cache.exists():
            feats = torch.load(cache, weights_only=False); info = feats["info"]
        else:
            encoder, info = load_encoder(path, ref, device)
            with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
                tl, t4, ty = extract(encoder, loader(train_set), device)
                vl, v4, vy = extract(encoder, loader(val_set), device)
            del encoder; torch.cuda.empty_cache()
            feats = {"info": info, "train": (tl, t4, ty), "val": (vl, v4, vy)}
            torch.save(feats, cache)
        extract_s = round(time.perf_counter() - start, 1)
        log(json.dumps({"name": name, "feature_extraction_seconds": extract_s, "train": len(feats["train"][2]), "val": len(feats["val"][2])}))
        (tl, t4, ty), (vl, v4, vy) = feats["train"], feats["val"]
        on = lambda t: t.to(device)
        features = {"last": (on(tl), on(ty), on(vl), on(vy)), "last4": (on(t4), on(ty), on(v4), on(vy))}
        num_classes = int(ty.max().item()) + 1
        res = sweep(features, device, num_classes, log, tuple(args.ref_lrs))
        res.update(info=info, seconds_total=round(time.perf_counter() - start, 1), feature_cache=str(cache))
        results[name] = res
        log(json.dumps({"name": name, "selected": res["selected"], "best_on_val": res["best_on_val"]}))
        del features; torch.cuda.empty_cache()
        out.write_text(json.dumps({"protocol": {
            "dataset": "ImageNet-100 (clane9, 160 px short side); train 126,689 / validation 5,000",
            "features": "target encoder; last = avg-pooled final-norm tokens; last4 = concat of avg-pooled final-norm outputs of the last 4 blocks",
            "head": "BatchNorm1d(affine=False) + Linear", "optimizer": "LARS, momentum 0.9, batch 16384, 50 epochs, lr/10 every 15 epochs",
            "sweep": {"ref_lr (x batch/256)": list(args.ref_lrs), "weight_decay": WEIGHT_DECAYS},
            "selection": "selected = best on held-out 10% of train, retrained on full train; best_on_val = paper-style max over sweep",
            "deviation": "features from eval transform (no probe-time augmentation)"}, "results": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
