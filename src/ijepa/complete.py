"""Masked-image completion with a frozen I-JEPA and either a masked-patch decoder or the RCDM visualizer.

    python -m src.ijepa.complete --image photo.jpg --encoder ENC.pt --decoder DEC.pt --output-dir out/ \\
        [--rect X0 Y0 X1 Y1 [--rect-space original|crop]] [--seed 0] [--device auto|cuda|cpu] [--context-tokens N]
    python -m src.ijepa.complete --image photo.jpg --encoder ENC.pt --rcdm RCDM.pt --output-dir out/ \\
        [--samples 4] [--steps 50] [--rcdm-conditioning predicted|visible|oracle] [--vae-cache DIR] [mask options]

- Preprocessing is the evaluation transform used in training: RGB, Resize(image_size + 32) on the short side,
  CenterCrop(image_size), ImageNet normalization.
- Masks: random multi-block masks sampled exactly as in training (encoder checkpoint's mask config, seeded), or one
  user rectangle. A rectangle in original-image pixels is mapped through the same resize/crop; every patch it
  touches is masked. The context is then every other patch.
- The decoder sees only the predicted target tokens (no visible pixels); visible patches are copied unchanged.
- RCDM (`--rcdm`, not retrained for inpainting): the pooled feature it was trained on (mean of all target-encoder
  tokens of the full image) is rebuilt from visible pixels only: target-encoder tokens of the visible patches plus
  predictor tokens for the hidden ones (`rcdm_condition`). DDIM then generates only the hidden latent cells, the
  visible cells being reset to the VAE latents of the masked image at every step (`rcdm.inpaint`). Several samples
  show what the features pin down (shared across samples) and what they do not (varies).
- Outputs: original.png (the 224 crop), masked.png, completed.png (RCDM: completed_1..N.png), comparison.png,
  metadata.json.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import IMAGENET_MEAN, IMAGENET_STD
from .decoder import MaskedPatchDecoder, compose_masked_images, gather_target_patches, patchify
from .masks import MaskBatch, MultiBlockMaskSampler
from .train_decoder import load_frozen_ijepa


class CompletionError(ValueError):
    """A user-facing problem with the inputs (mismatched checkpoints, bad rectangle, ...)."""


# ---------------------------------------------------------------------------------------------------------- images
def crop_geometry(width: int, height: int, image_size: int) -> dict:
    """Geometry of Resize(image_size + 32) then CenterCrop(image_size), using torchvision's integer rules."""
    target = image_size + 32
    if width <= height:
        new_w, new_h = target, int(target * height / width)
    else:
        new_w, new_h = int(target * width / height), target
    top = int(round((new_h - image_size) / 2.0))
    left = int(round((new_w - image_size) / 2.0))
    return {"original_size": [width, height], "resized_size": [new_w, new_h], "crop_left": left, "crop_top": top,
            "crop_size": image_size, "scale_x": new_w / width, "scale_y": new_h / height}


def preprocess(image, image_size: int) -> tuple[torch.Tensor, dict]:
    """PIL image -> normalized tensor [1,3,S,S] (the training eval transform) plus its crop geometry."""
    from torchvision import transforms

    image = image.convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(image_size + 32), transforms.CenterCrop(image_size),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transform(image).unsqueeze(0), crop_geometry(image.width, image.height, image_size)


def denormalize(images: torch.Tensor) -> torch.Tensor:
    mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0, 1)


# ----------------------------------------------------------------------------------------------------------- masks
def rect_to_patch_indices(rect, geometry: dict, patch_size: int, space: str = "original") -> list[int]:
    """Row-major indices of every patch a rectangle (x0, y0, x1, y1; x1/y1 exclusive) touches in the crop."""
    x0, y0, x1, y1 = (float(v) for v in rect)
    if not (x1 > x0 and y1 > y0):
        raise CompletionError(f"rectangle needs x1 > x0 and y1 > y0, got {rect}")
    size = geometry["crop_size"]
    if space == "original":
        x0, x1 = x0 * geometry["scale_x"] - geometry["crop_left"], x1 * geometry["scale_x"] - geometry["crop_left"]
        y0, y1 = y0 * geometry["scale_y"] - geometry["crop_top"], y1 * geometry["scale_y"] - geometry["crop_top"]
    elif space != "crop":
        raise CompletionError(f"unknown rectangle space: {space}")
    x0, y0, x1, y1 = max(x0, 0.0), max(y0, 0.0), min(x1, float(size)), min(y1, float(size))
    if x1 <= x0 or y1 <= y0:
        raise CompletionError("rectangle lies entirely outside the centre crop the model sees")
    grid = size // patch_size
    cols = range(int(math.floor(x0 / patch_size)), int(math.ceil(x1 / patch_size)))
    rows = range(int(math.floor(y0 / patch_size)), int(math.ceil(y1 / patch_size)))
    return [r * grid + c for r in rows for c in cols]


def rect_mask_batch(target_indices: list[int], num_patches: int, context_tokens: int | None = None,
                    seed: int = 0) -> MaskBatch:
    """One target block = the rectangle's patches; context = every other patch (optionally subsampled)."""
    targets = sorted(set(target_indices))
    target_set = set(targets)
    context = [i for i in range(num_patches) if i not in target_set]
    if not context:
        raise CompletionError("the rectangle covers the whole image; at least one visible patch is required")
    context = _subsample(context, context_tokens, seed)
    return MaskBatch(torch.tensor([context]), torch.tensor([[targets]]))


def random_mask_batch(mask_cfg: dict, image_size: int, patch_size: int, seed: int,
                      context_tokens: int | None = None) -> MaskBatch:
    sampler = MultiBlockMaskSampler(**{**mask_cfg, "image_size": image_size, "patch_size": patch_size})
    masks = sampler.sample(1, torch.Generator().manual_seed(seed))
    context = _subsample(masks.context[0].tolist(), context_tokens, seed)
    return MaskBatch(torch.tensor([context]), masks.targets)


def _subsample(context: list[int], context_tokens: int | None, seed: int) -> list[int]:
    if not context_tokens or context_tokens >= len(context):
        return context
    order = torch.randperm(len(context), generator=torch.Generator().manual_seed(seed + 1))[:context_tokens]
    return sorted(context[i] for i in order.tolist())


# ---------------------------------------------------------------------------------------------------------- models
def load_models(encoder_path: str, decoder_path: str, device: torch.device, allow_mismatch: bool = False,
                allow_oracle: bool = False):
    """Load the frozen I-JEPA and its decoder, refusing pairs that were not trained together."""
    encoder, enc_ckpt = load_frozen_ijepa(encoder_path, device)
    dec_ckpt = torch.load(decoder_path, map_location="cpu", weights_only=False)
    dec_cfg = dec_ckpt.get("config", {})
    if "decoder" not in dec_cfg:
        raise CompletionError(f"{decoder_path} is not a masked-patch decoder checkpoint (no 'decoder' config)")
    model_cfg = enc_ckpt["config"]["model"]
    encoder_dim = int(model_cfg["encoder_dim"])
    trained_dim = int(dec_ckpt["model"]["input_proj.weight"].shape[1])
    if trained_dim != encoder_dim:
        raise CompletionError(f"decoder expects {trained_dim}-dim encoder tokens but the encoder produces {encoder_dim}")
    conditioning = dec_cfg.get("conditioning", "predicted")
    if conditioning != "predicted" and not allow_oracle:
        raise CompletionError(f"decoder was trained with conditioning={conditioning!r}, which reads the hidden region; "
                              "it cannot complete real images (pass --allow-oracle only for diagnostics)")
    trained_with = Path(str(dec_cfg.get("encoder_checkpoint", ""))).name
    if trained_with and trained_with != Path(encoder_path).name and not allow_mismatch:
        raise CompletionError(f"decoder was trained with encoder '{trained_with}', not '{Path(encoder_path).name}'. "
                              "A decoder only reads the features of the encoder it was trained with "
                              "(pass --allow-mismatch to override)")
    decoder = MaskedPatchDecoder(image_size=int(model_cfg["image_size"]), patch_size=int(model_cfg["patch_size"]),
                                 encoder_dim=encoder_dim, **dec_cfg["decoder"])
    decoder.load_state_dict(dec_ckpt["model"])
    decoder.requires_grad_(False).eval()
    info = {"encoder_checkpoint": str(encoder_path), "encoder_step": enc_ckpt.get("state", {}).get("global_step"),
            "decoder_checkpoint": str(decoder_path), "decoder_step": dec_ckpt.get("state", {}).get("global_step"),
            "decoder_trained_with_encoder": trained_with or None, "conditioning": conditioning}
    return encoder, decoder.to(device), enc_ckpt["config"], info


@torch.no_grad()
def complete(encoder, decoder, image: torch.Tensor, masks: MaskBatch, patch_size: int, device: torch.device) -> dict:
    image, masks = image.to(device), masks.to(device)
    with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
        tokens = encoder.predictions(image, masks)
        patches = decoder(tokens, masks.targets)
    patches = patches.float()
    masked, completed, target_mask = compose_masked_images(image, patches, masks.targets, patch_size)
    truth = gather_target_patches(patchify(image, patch_size), masks.targets)
    visible = ~target_mask[0].bool()
    image_patches, completed_patches = patchify(image, patch_size)[0], patchify(completed, patch_size)[0]
    return {"masked": masked, "completed": completed, "target_mask": target_mask,
            "masked_l1": float((patches - truth).abs().mean()), "masked_mse": float((patches - truth).pow(2).mean()),
            "visible_pixels_exact": bool(torch.equal(image_patches[visible], completed_patches[visible])),
            "masked_patches": int(target_mask.sum()), "context_patches": int(masks.context.size(1))}


# ------------------------------------------------------------------------------------------------------------ RCDM
RCDM_CONDITIONING = ("predicted", "visible", "oracle")


def load_rcdm(path: str, encoder_dim: int, device: torch.device, conditioning: str = "predicted",
              allow_oracle: bool = False):
    """The EMA UNet of an RCDM checkpoint (train_rcdm.py), refusing feature-width mismatches and silent oracles."""
    from .train_rcdm import RCDMState

    if conditioning not in RCDM_CONDITIONING:
        raise CompletionError(f"unknown RCDM conditioning {conditioning!r}; expected one of {RCDM_CONDITIONING}")
    if conditioning == "oracle" and not allow_oracle:
        raise CompletionError("RCDM conditioning 'oracle' reads the hidden region; it cannot complete real images "
                              "(pass --allow-oracle only for diagnostics)")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("config", {})
    if "diffusion" not in cfg or "model" not in cfg:
        raise CompletionError(f"{path} is not an RCDM checkpoint (no 'diffusion' config)")
    feature_dim = int(cfg["model"]["feature_dim"])
    if feature_dim != encoder_dim:
        raise CompletionError(f"RCDM expects {feature_dim}-dim features but the encoder produces {encoder_dim}")
    state = RCDMState(cfg["model"])
    state.load_state_dict(checkpoint["model"])
    info = {"rcdm_checkpoint": str(path), "rcdm_step": checkpoint.get("state", {}).get("global_step"),
            "rcdm_weights": "EMA UNet", "rcdm_conditioning": conditioning}
    return state.ema.requires_grad_(False).eval().to(device), info


def hidden_patches(masks: MaskBatch) -> list[int]:
    return sorted(set(masks.targets.flatten().tolist()))


@torch.no_grad()
def rcdm_condition(encoder, image: torch.Tensor, masks: MaskBatch, num_patches: int, mode: str) -> torch.Tensor:
    """Pooled feature [1,D] for the RCDM, which was trained on mean_p T(x)_p over every token of the full image.

    predicted (deployable): the target encoder T on the visible patches only, plus predictor tokens for the hidden
        patches. The predictor learned per-token LayerNorm'ed targets (`IJEPAModel.targets`), so its tokens are put
        back on the scale of T with the mean and standard deviation of the visible tokens.
    visible: the mean of the visible-only tokens (the hole is ignored).
    oracle: the full-image feature. It reads the hidden region: a diagnostic upper bound only.
    """
    if mode == "oracle":
        return encoder.target_encoder(image).mean(1).float()
    hidden = set(hidden_patches(masks))
    visible = torch.tensor([[p for p in range(num_patches) if p not in hidden]], device=image.device)
    tokens = encoder.target_encoder(image, visible).float()            # [1,V,D]; patch embedding is per patch
    if mode == "visible":
        return tokens.mean(1)
    predicted = encoder.predictions(image, masks).float()              # [M,1,Nt,D]
    total = predicted.new_zeros(num_patches, predicted.size(-1))
    count = predicted.new_zeros(num_patches, 1)
    for block, indices in zip(predicted[:, 0], masks.targets[:, 0]):
        total.index_add_(0, indices, block)
        count.index_add_(0, indices, torch.ones_like(block[:, :1]))
    rows = count[:, 0] > 0
    hole = total[rows] / count[rows]                                   # overlapping blocks averaged
    mean, std = tokens.mean(-1).mean(), tokens.var(-1, unbiased=False).sqrt().mean()
    return (tokens.sum(1) + (hole * std + mean).sum(0, keepdim=True)) / num_patches


@torch.no_grad()
def complete_rcdm(encoder, rcdm, vae, image: torch.Tensor, masks: MaskBatch, patch_size: int, device: torch.device,
                  conditioning: str = "predicted", samples: int = 4, steps: int = 50, seed: int = 0,
                  resample: int = 1) -> dict:
    from .rcdm import decode_latents, encode_latents, inpaint

    image, masks = image.to(device), masks.to(device)
    size = image.size(-1)
    grid = size // patch_size
    hidden = hidden_patches(masks)
    patch_mask = torch.zeros(grid * grid, device=device)
    patch_mask[hidden] = 1
    pixel_mask = patch_mask.view(1, 1, grid, grid).repeat_interleave(patch_size, 2).repeat_interleave(patch_size, 3)
    with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):  # as the RCDM cache was built
        feature = rcdm_condition(encoder, image, masks, grid * grid, conditioning)
        full = encoder.target_encoder(image).mean(1).float()           # only to report how close `feature` is
    masked = image * (1 - pixel_mask)                                  # hidden pixels -> 0 = ImageNet mean colour
    known = encode_latents(vae, masked).float()                        # the true hidden pixels never reach the VAE
    keep = 1 - F.max_pool2d(pixel_mask, size // known.size(-1))        # latent cells touching the hole are generated
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = inpaint(rcdm, feature.expand(samples, -1), known.expand(samples, -1, -1, -1), keep, steps=steps,
                      generator=generator, resample=resample)
    generated = decode_latents(vae, latents)                           # [N,3,S,S] in [0,1]
    mean = generated.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = generated.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    hole = pixel_mask.bool().expand(samples, 3, size, size)
    truth = image.expand(samples, -1, -1, -1)
    completed = torch.where(hole, (generated - mean) / std, truth)
    l1 = [float((completed[i] - truth[i]).abs()[hole[i]].mean()) for i in range(samples)]
    mse = [float((completed[i] - truth[i]).pow(2)[hole[i]].mean()) for i in range(samples)]
    return {"masked": masked, "completed": completed,
            "masked_l1": sum(l1) / samples, "masked_mse": sum(mse) / samples,
            "masked_l1_per_sample": l1, "masked_mse_per_sample": mse,
            "visible_pixels_exact": bool(torch.equal(completed[~hole], truth[~hole])),
            "masked_patches": len(hidden), "context_patches": int(masks.context.size(1)),
            "conditioning_cosine_to_full_image_feature": float(F.cosine_similarity(feature, full).item())}


# ------------------------------------------------------------------------------------------------------------- CLI
def run(args) -> dict:
    from PIL import Image
    from torchvision.utils import save_image

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise CompletionError("--device cuda requested but CUDA is not available; use --device cpu")
    if args.rcdm:
        from .rcdm import load_vae

        encoder, enc_ckpt = load_frozen_ijepa(args.encoder, device)
        enc_cfg = enc_ckpt["config"]
        rcdm, rcdm_info = load_rcdm(args.rcdm, int(enc_cfg["model"]["encoder_dim"]), device, args.rcdm_conditioning,
                                    args.allow_oracle)
        info = {"encoder_checkpoint": str(args.encoder), "encoder_step": enc_ckpt.get("state", {}).get("global_step"),
                **rcdm_info, "samples": args.samples, "ddim_steps": args.steps, "resample": args.resample}
        vae = load_vae(device, cache_dir=args.vae_cache)
    else:
        encoder, decoder, enc_cfg, info = load_models(args.encoder, args.decoder, device, args.allow_mismatch,
                                                      args.allow_oracle)
    image_size, patch_size = int(enc_cfg["model"]["image_size"]), int(enc_cfg["model"]["patch_size"])
    num_patches = (image_size // patch_size) ** 2
    image, geometry = preprocess(Image.open(args.image), image_size)
    if args.rect:
        targets = rect_to_patch_indices(args.rect, geometry, patch_size, args.rect_space)
        masks = rect_mask_batch(targets, num_patches, args.context_tokens, args.seed)
        mask_mode = {"mode": "rect", "rect": list(args.rect), "rect_space": args.rect_space}
    else:
        masks = random_mask_batch(enc_cfg["mask"], image_size, patch_size, args.seed, args.context_tokens)
        mask_mode = {"mode": "random_multiblock", "num_targets": int(masks.targets.size(0))}
    try:
        if args.rcdm:
            result = complete_rcdm(encoder, rcdm, vae, image, masks, patch_size, device, args.rcdm_conditioning,
                                   args.samples, args.steps, args.seed, args.resample)
        else:
            result = complete(encoder, decoder, image, masks, patch_size, device)
    except torch.cuda.OutOfMemoryError as error:  # pragma: no cover - hardware dependent
        raise CompletionError(f"CUDA out of memory ({error}); rerun with --device cpu") from error

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    original, masked, completed = (denormalize(t.float().cpu()) for t in (image, result["masked"], result["completed"]))
    save_image(original, out / "original.png")
    save_image(masked, out / "masked.png")
    if args.rcdm:
        for index, sample_image in enumerate(completed, start=1):
            save_image(sample_image, out / f"completed_{index}.png")
    else:
        save_image(completed, out / "completed.png")
    save_image(torch.cat([original, masked, completed]), out / "comparison.png", nrow=2 + len(completed), padding=4,
               pad_value=1.0)
    meta = {**info, "image": str(args.image), "device": str(device), "seed": args.seed, **mask_mode,
            "context_tokens_requested": args.context_tokens, "geometry": geometry,
            **{k: v for k, v in result.items() if k not in ("masked", "completed", "target_mask")},
            "units": "masked_l1/mse are in ImageNet-normalized pixel units over the masked patches"
                     + (" (RCDM: over the union of hidden pixels, mean of the samples)" if args.rcdm else ""),
            "target_patch_indices": sorted(set(masks.targets.flatten().tolist()))}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Complete masked regions of an image with I-JEPA + decoder or RCDM.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--encoder", required=True, help="I-JEPA checkpoint (full or epoch_*_weights.pt)")
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--decoder", help="masked-patch decoder checkpoint (e.g. best.pt)")
    model.add_argument("--rcdm", help="RCDM checkpoint from train_rcdm.py (its EMA UNet is used)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rect", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                        help="mask this rectangle (x1/y1 exclusive) instead of random multi-block masks")
    parser.add_argument("--rect-space", choices=("original", "crop"), default="original",
                        help="coordinates of --rect: original image pixels (default) or the model's centre crop")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cuda or cpu")
    parser.add_argument("--context-tokens", type=int, default=None,
                        help="subsample the visible context to N tokens (training batches saw ~60 at batch 64)")
    parser.add_argument("--allow-mismatch", action="store_true", help="allow a decoder trained with another encoder")
    parser.add_argument("--allow-oracle", action="store_true", help="allow oracle-conditioned decoders (diagnostic)")
    parser.add_argument("--samples", type=int, default=4, help="RCDM: completions to draw")
    parser.add_argument("--steps", type=int, default=50, help="RCDM: DDIM steps")
    parser.add_argument("--resample", type=int, default=1,
                        help="RCDM: RePaint repeats per step (1 = plain blending; more harmonises seams, slower)")
    parser.add_argument("--rcdm-conditioning", choices=RCDM_CONDITIONING, default="predicted",
                        help="RCDM feature: visible tokens + predictor (default), visible tokens only, or oracle")
    parser.add_argument("--vae-cache", default=None, help="RCDM: Hugging Face cache dir holding sd-vae-ft-mse")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    try:
        meta = run(args)
    except CompletionError as error:
        raise SystemExit(f"error: {error}") from None
    print(json.dumps({k: meta[k] for k in ("masked_patches", "context_patches", "masked_l1", "visible_pixels_exact",
                                           "conditioning_cosine_to_full_image_feature") if k in meta}))


if __name__ == "__main__":
    main()
