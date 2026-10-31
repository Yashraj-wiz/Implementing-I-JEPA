"""Side-by-side gallery: masked-patch decoder vs RCDM completions of the same images and masks (CLI outputs).

Each row: original | masked | decoder completion | RCDM samples. Both methods are scored the same way here:
mean absolute error over the hidden pixels in [0, 1] RGB, computed from the saved 8-bit PNGs (the CLI's own
`masked_l1` is in ImageNet-normalized units and, for the decoder, averages overlapping blocks). The script
checks that both runs hid exactly the same patches.

    python scripts/rcdm_cli_gallery.py --decoder-dir artifacts/cli_demo_vitl --rcdm-dir artifacts/cli_demo_rcdm         [--extra "STL-10 ViT-S decoder=artifacts/cli_demo_stl10/stl10_decoder"]   # more deterministic columns
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def hole_mask(indices: list[int], size: int, patch: int) -> np.ndarray:
    grid = size // patch
    mask = np.zeros((grid, grid), dtype=bool)
    for index in indices:
        mask[index // grid, index % grid] = True
    return mask.repeat(patch, 0).repeat(patch, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--decoder-dir", default="artifacts/cli_demo_vitl")
    parser.add_argument("--rcdm-dir", default="artifacts/cli_demo_rcdm")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--extra", action="append", default=[], metavar="NAME=DIR",
                        help="another deterministic decoder's CLI output folder, shown after decoder v1 (repeatable)")
    args = parser.parse_args()
    extras = [tuple(spec.split("=", 1)) for spec in args.extra]
    rcdm_dir, decoder_dir = Path(args.rcdm_dir), Path(args.decoder_dir)

    rows, records = [], []
    for case in sorted(p for p in rcdm_dir.iterdir() if (p / "metadata.json").exists()):
        rcdm_meta = json.loads((case / "metadata.json").read_text(encoding="utf-8"))
        dec_meta = json.loads((decoder_dir / case.name / "metadata.json").read_text(encoding="utf-8"))
        if rcdm_meta["target_patch_indices"] != dec_meta["target_patch_indices"]:
            raise SystemExit(f"{case.name}: the two runs hid different patches")
        original = load(case / "original.png")
        mask = hole_mask(rcdm_meta["target_patch_indices"], original.shape[0], args.patch_size)
        decoder = load(decoder_dir / case.name / "completed.png")
        extra_images = []
        for name, folder in extras:
            extra_meta = json.loads((Path(folder) / case.name / "metadata.json").read_text(encoding="utf-8"))
            if extra_meta["target_patch_indices"] != rcdm_meta["target_patch_indices"]:
                raise SystemExit(f"{case.name}: {name} hid different patches")
            extra_images.append(load(Path(folder) / case.name / "completed.png"))
        samples = [load(case / f"completed_{i}.png") for i in range(1, args.samples + 1)]
        l1 = lambda image: float(np.abs(image - original)[mask].mean())
        record = {"case": case.name, "hidden_patches": len(rcdm_meta["target_patch_indices"]),
                  "decoder_l1": round(l1(decoder), 4), "rcdm_l1": [round(l1(s), 4) for s in samples],
                  "rcdm_conditioning_cosine": round(rcdm_meta["conditioning_cosine_to_full_image_feature"], 4),
                  "visible_pixels_exact": rcdm_meta["visible_pixels_exact"]}
        for (name, _), image in zip(extras, extra_images):
            record[f"{name}_l1"] = round(l1(image), 4)
        record["rcdm_l1_mean"] = round(float(np.mean(record["rcdm_l1"])), 4)
        records.append(record)
        extra_text = "".join(f"  {name} {record[f'{name}_l1']:.3f}" for name, _ in extras)
        rows.append(([original, load(case / "masked.png"), decoder, *extra_images, *samples],
                     f"{case.name}  decoder {record['decoder_l1']:.3f}{extra_text}  RCDM {record['rcdm_l1_mean']:.3f}  "
                     f"cos {record['rcdm_conditioning_cosine']:.3f}"))

    size, pad, band = rows[0][0][0].shape[0], 4, 14
    columns = len(rows[0][0])
    canvas = Image.new("RGB", (columns * (size + pad) + pad, len(rows) * (size + pad + band) + pad + band), "white")
    draw = ImageDraw.Draw(canvas)
    names = "".join(f" | {name}" for name, _ in extras)
    header = f"original | masked | decoder v1{names} | RCDM samples 1-{args.samples}   (L1 = mean |error| over hidden pixels, RGB in [0,1])"
    draw.text((pad, 2), header, fill="black")
    for r, (tiles, caption) in enumerate(rows):
        y = band + pad + r * (size + pad + band)
        draw.text((pad, y), caption, fill="black")
        for c, tile in enumerate(tiles):
            canvas.paste(Image.fromarray((tile * 255).round().astype(np.uint8)), (pad + c * (size + pad), y + band))
    canvas.save(rcdm_dir / "gallery.png")
    summary = {"cases": records,
               "decoder_l1_mean": round(float(np.mean([r["decoder_l1"] for r in records])), 4),
               "rcdm_l1_mean": round(float(np.mean([r["rcdm_l1_mean"] for r in records])), 4),
               **{f"{name}_l1_mean": round(float(np.mean([r[f"{name}_l1"] for r in records])), 4) for name, _ in extras},
               "metric": "mean absolute error over hidden pixels, RGB in [0,1], from the saved PNGs"}
    (rcdm_dir / "gallery.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
