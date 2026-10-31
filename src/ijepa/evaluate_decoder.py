from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .data import make_dataset
from .decoder import MaskedPatchDecoder, gather_target_patches, patchify
from .masks import MultiBlockMaskSampler
from .train_decoder import decoder_conditioning, load_frozen_ijepa, save_sample_grid


@torch.inference_mode()
def evaluate_pairing(
    name: str,
    encoder_path: str,
    decoder_path: str,
    output_dir: Path,
    data_root: str,
    examples: int,
    batch_size: int,
    mask_seed: int,
    conditioning: str | None = None,
    shuffle_latents: bool = False,
    num_workers: int = 8,
    cache_dir: str | None = None,
) -> dict:
    """Fixed-mask masked L1/MSE for one encoder/decoder pairing.

    ``conditioning`` defaults to the mode the decoder was trained with. With
    ``shuffle_latents`` every image receives the latents of the next image in the batch
    (a derangement), which measures how much the decoder relies on its latent input.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, encoder_checkpoint = load_frozen_ijepa(encoder_path, device)
    encoder_cfg = encoder_checkpoint["config"]
    model_cfg = encoder_cfg["model"]

    decoder_checkpoint = torch.load(decoder_path, map_location="cpu", weights_only=False)
    decoder = MaskedPatchDecoder(
        image_size=int(model_cfg["image_size"]),
        patch_size=int(model_cfg["patch_size"]),
        encoder_dim=int(model_cfg["encoder_dim"]),
        **decoder_checkpoint["config"]["decoder"],
    ).to(device)
    decoder.load_state_dict(decoder_checkpoint["model"])
    decoder.eval()
    if conditioning is None:
        conditioning = decoder_checkpoint["config"].get("conditioning", "predicted")
    normalize_targets = bool(encoder_cfg["objective"].get("normalize_targets", True))

    data_cfg = {**encoder_cfg["data"], "root": data_root}
    if cache_dir:  # e.g. the local Hugging Face Arrow cache instead of the checkpoint's /vol/hf_cache
        data_cfg["cache_dir"] = cache_dir
    dataset = make_dataset(data_cfg, train=False, split="test")
    dataset = Subset(dataset, range(min(examples, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    mask_cfg = {
        **encoder_cfg["mask"],
        "image_size": int(model_cfg["image_size"]),
        "patch_size": int(model_cfg["patch_size"]),
    }
    mask_sampler = MultiBlockMaskSampler(**mask_cfg)
    mask_generator = torch.Generator().manual_seed(mask_seed)
    patch_size = int(model_cfg["patch_size"])
    absolute_error = 0.0
    squared_error = 0.0
    elements = 0

    for batch_index, (images, _labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks = mask_sampler.sample(images.size(0), mask_generator).to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            predicted_tokens = decoder_conditioning(
                encoder, images, masks, conditioning, normalize_targets
            )
            if shuffle_latents:
                predicted_tokens = predicted_tokens.roll(1, dims=1)
            predicted_patches = decoder(predicted_tokens, masks.targets)
            target_patches = gather_target_patches(
                patchify(images, patch_size), masks.targets
            )
        absolute_error += float(
            F.l1_loss(predicted_patches.float(), target_patches.float(), reduction="sum")
        )
        squared_error += float(
            F.mse_loss(predicted_patches.float(), target_patches.float(), reduction="sum")
        )
        elements += predicted_patches.numel()
        if batch_index == 0:
            save_sample_grid(
                output_dir / f"{name}.png",
                images,
                predicted_patches,
                masks.targets,
                patch_size,
            )

    result = {
        "name": name,
        "encoder_checkpoint": encoder_path,
        "encoder_step": int(encoder_checkpoint["state"]["global_step"]),
        "decoder_checkpoint": decoder_path,
        "decoder_step": int(decoder_checkpoint["state"]["global_step"]),
        "conditioning": conditioning,
        "shuffle_latents": shuffle_latents,
        "examples": len(dataset),
        "mask_seed": mask_seed,
        "masked_l1": absolute_error / elements,
        "masked_mse": squared_error / elements,
        "sample_grid": str(output_dir / f"{name}.png"),
    }
    del decoder, encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def evaluate_decoder_pairings(
    output_dir: str,
    data_root: str = "/vol/data",
    examples: int = 1024,
    batch_size: int = 128,
    mask_seed: int = 20260831,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    encoder_58k = "/vol/encoder_checkpoints/local_step_0058000.pt"
    encoder_200k = "/vol/runs/stl10_vits16_continuation_b128/step_0200000.pt"
    decoder_original = "/vol/runs/stl10_masked_decoder_300k_step58000/step_0300000.pt"
    decoder_adapted = "/vol/runs/stl10_masked_decoder_final_encoder_adapt_50k/step_0050000.pt"
    pairings = [
        ("paired_step58k_original", encoder_58k, decoder_original),
        ("mismatched_step200k_original", encoder_200k, decoder_original),
        ("paired_step200k_adapted", encoder_200k, decoder_adapted),
    ]
    results = {
        name: evaluate_pairing(
            name,
            encoder_path,
            decoder_path,
            output,
            data_root,
            examples,
            batch_size,
            mask_seed,
        )
        for name, encoder_path, decoder_path in pairings
    }
    payload = {
        "protocol": {
            "dataset": "STL-10 test",
            "examples": examples,
            "batch_size": batch_size,
            "mask_seed": mask_seed,
            "metrics": "elementwise error over all predicted target patches",
        },
        "results": results,
    }
    report = output / "decoder_controlled_evaluation.json"
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload), flush=True)
    return payload


def main() -> None:
    """Local entry point: evaluate named decoder pairings on a fixed benchmark (the encoder's dataset, "test" split)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True)
    parser.add_argument(
        "--decoder", action="append", required=True, metavar="NAME=PATH",
        help="repeatable; conditioning is read from each decoder checkpoint",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--cache-dir", default=None, help="Hugging Face cache for imagenet100 (overrides the checkpoint's)")
    parser.add_argument("--examples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--mask-seed", type=int, default=20260831)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--with-shuffled", action="store_true", help="also evaluate with latents shuffled across images")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for spec in args.decoder:
        name, path = spec.split("=", 1)
        for shuffled in (False, True) if args.with_shuffled else (False,):
            key = f"{name}_shuffled" if shuffled else name
            results[key] = evaluate_pairing(
                key, args.encoder, path, output, args.data_root, args.examples,
                args.batch_size, args.mask_seed, shuffle_latents=shuffled,
                num_workers=args.num_workers, cache_dir=args.cache_dir,
            )
            print(json.dumps({k: results[key][k] for k in ("name", "conditioning", "masked_l1", "masked_mse")}), flush=True)
    payload = {
        "protocol": {
            "dataset": f"{_dataset_name(args.encoder)} test split", "examples": args.examples, "batch_size": args.batch_size,
            "mask_seed": args.mask_seed, "metrics": "elementwise error over all predicted target patches",
        },
        "results": results,
    }
    (output / "decoder_evaluation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _dataset_name(encoder_path: str) -> str:
    checkpoint = torch.load(encoder_path, map_location="cpu", weights_only=False)
    return str(checkpoint["config"]["data"]["name"])


if __name__ == "__main__":
    main()
