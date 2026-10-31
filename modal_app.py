from pathlib import Path

import modal


PROJECT = "/root/project"
VOLUME_ROOT = "/vol"
app = modal.App("ijepa-research")
volume = modal.Volume.from_name("ijepa-research", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "pyyaml==6.0.2", "numpy==2.1.3")
    .add_local_dir("src", f"{PROJECT}/src")
    .add_local_dir("configs", f"{PROJECT}/configs")
)


@app.function(image=image, volumes={VOLUME_ROOT: volume}, timeout=60 * 30)
def prepare_stl10():
    from torchvision.datasets import STL10

    root = Path(VOLUME_ROOT) / "data"
    STL10(root=str(root), split="unlabeled", download=True)
    volume.commit()
    return {"dataset": str(root), "files": sum(1 for p in root.rglob("*") if p.is_file())}


@app.function(
    image=image,
    gpu="L4",
    volumes={VOLUME_ROOT: volume},
    timeout=60 * 30,
    retries=modal.Retries(max_retries=1, initial_delay=0.0),
)
def smoke_gpu():
    import sys
    import torch

    sys.path.insert(0, PROJECT)
    from src.ijepa.train import train

    result = train(f"{PROJECT}/configs/smoke.yaml")
    return {**result, "gpu": torch.cuda.get_device_name(), "memory_gb": torch.cuda.get_device_properties(0).total_memory / 2**30}


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={VOLUME_ROOT: volume},
    timeout=60 * 60 * 24,
    retries=modal.Retries(max_retries=10, initial_delay=0.0),
)
def pretrain_stl10():
    import sys
    import torch

    sys.path.insert(0, PROJECT)
    from src.ijepa.train import train

    result = train(f"{PROJECT}/configs/stl10_vits16.yaml", commit_callback=volume.commit)
    return {**result, "gpu": torch.cuda.get_device_name()}


@app.function(
    image=image,
    gpu="L4",
    volumes={VOLUME_ROOT: volume},
    timeout=60 * 60 * 24,
    retries=modal.Retries(max_retries=10, initial_delay=0.0),
)
def continue_stl10_l4():
    import sys
    import torch

    sys.path.insert(0, PROJECT)
    from src.ijepa.train import train

    result = train(
        f"{PROJECT}/configs/stl10_vits16_l4_continuation.yaml",
        commit_callback=volume.commit,
    )
    return {**result, "gpu": torch.cuda.get_device_name()}


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={VOLUME_ROOT: volume},
    timeout=60 * 30,
    retries=modal.Retries(max_retries=1, initial_delay=0.0),
)
def profile_stl10():
    import sys
    import torch

    sys.path.insert(0, PROJECT)
    from src.ijepa.train import train

    result = train(f"{PROJECT}/configs/stl10_vits16_profile.yaml", commit_callback=volume.commit)
    return {**result, "gpu": torch.cuda.get_device_name(), "memory_gb": torch.cuda.get_device_properties(0).total_memory / 2**30}


@app.function(
    image=image,
    gpu="L4",
    volumes={VOLUME_ROOT: volume},
    timeout=60 * 30,
    retries=modal.Retries(max_retries=1, initial_delay=0.0),
)
def profile_stl10_l4():
    import sys
    import torch

    sys.path.insert(0, PROJECT)
    from src.ijepa.train import train

    result = train(f"{PROJECT}/configs/stl10_vits16_l4_profile.yaml", commit_callback=volume.commit)
    return {**result, "gpu": torch.cuda.get_device_name(), "memory_gb": torch.cuda.get_device_properties(0).total_memory / 2**30}


@app.function(
    image=image,
    gpu="L4",
    volumes={VOLUME_ROOT: volume},
    timeout=60 * 30,
    retries=modal.Retries(max_retries=1, initial_delay=0.0),
)
def profile_stl10_l4_b256():
    import sys
    import torch

    sys.path.insert(0, PROJECT)
    from src.ijepa.train import train

    result = train(
        f"{PROJECT}/configs/stl10_vits16_l4_b256_profile.yaml",
        commit_callback=volume.commit,
    )
    return {
        **result,
        "gpu": torch.cuda.get_device_name(),
        "memory_gb": torch.cuda.get_device_properties(0).total_memory / 2**30,
    }


@app.function(
    image=image,
    gpu="T4",
    volumes={VOLUME_ROOT: volume},
    timeout=60 * 30,
    retries=modal.Retries(max_retries=1, initial_delay=0.0),
)
def profile_stl10_t4():
    import sys
    import torch

    sys.path.insert(0, PROJECT)
    from src.ijepa.train import train

    result = train(
        f"{PROJECT}/configs/stl10_vits16_t4_profile.yaml",
        commit_callback=volume.commit,
    )
    return {
        **result,
        "gpu": torch.cuda.get_device_name(),
        "memory_gb": torch.cuda.get_device_properties(0).total_memory / 2**30,
    }


@app.function(
    image=image,
    gpu="L4",
    volumes={VOLUME_ROOT: volume},
    timeout=60 * 30,
    retries=modal.Retries(max_retries=1, initial_delay=0.0),
)
def profile_masked_decoder_l4():
    import sys
    import torch

    sys.path.insert(0, PROJECT)
    from src.ijepa.train_decoder import train_decoder

    result = train_decoder(
        f"{PROJECT}/configs/stl10_masked_decoder_l4_profile.yaml",
        commit_callback=volume.commit,
    )
    return {
        **result,
        "gpu": torch.cuda.get_device_name(),
        "memory_gb": torch.cuda.get_device_properties(0).total_memory / 2**30,
    }


@app.function(
    image=image,
    gpu="L4",
    volumes={VOLUME_ROOT: volume},
    timeout=60 * 60 * 24,
    retries=modal.Retries(max_retries=10, initial_delay=0.0),
)
def train_masked_decoder_l4():
    import sys
    import torch

    sys.path.insert(0, PROJECT)
    from src.ijepa.train_decoder import train_decoder

    result = train_decoder(
        f"{PROJECT}/configs/stl10_masked_decoder_l4_300k.yaml",
        commit_callback=volume.commit,
    )
    return {**result, "gpu": torch.cuda.get_device_name()}


@app.function(
    image=image,
    gpu="L4",
    volumes={VOLUME_ROOT: volume},
    timeout=60 * 60 * 8,
    retries=modal.Retries(max_retries=10, initial_delay=0.0),
)
def adapt_masked_decoder_final_encoder_l4():
    import sys
    import torch

    sys.path.insert(0, PROJECT)
    from src.ijepa.train_decoder import train_decoder

    result = train_decoder(
        f"{PROJECT}/configs/stl10_masked_decoder_final_encoder_adapt_50k.yaml",
        commit_callback=volume.commit,
    )
    return {**result, "gpu": torch.cuda.get_device_name()}


@app.function(
    image=image,
    gpu="L4",
    volumes={VOLUME_ROOT: volume},
    timeout=60 * 60,
)
def evaluate_stl10():
    import sys

    sys.path.insert(0, PROJECT)
    from src.ijepa.evaluate import evaluate_stl10 as run_evaluation

    checkpoints = {
        "step_058000": f"{VOLUME_ROOT}/encoder_checkpoints/local_step_0058000.pt",
        "step_200000": f"{VOLUME_ROOT}/runs/stl10_vits16_continuation_b128/step_0200000.pt",
    }
    result = {
        name: run_evaluation(path, f"{VOLUME_ROOT}/data")
        for name, path in checkpoints.items()
    }
    volume.commit()
    return result


@app.function(
    image=image,
    gpu="L4",
    volumes={VOLUME_ROOT: volume},
    timeout=60 * 30,
)
def evaluate_decoder_controlled_l4():
    import sys

    sys.path.insert(0, PROJECT)
    from src.ijepa.evaluate_decoder import evaluate_decoder_pairings

    result = evaluate_decoder_pairings(
        f"{VOLUME_ROOT}/evaluations/decoder_controlled_step58k_vs_final"
    )
    volume.commit()
    return result


# ---------------------------------------------------------------------------
# ImageNet-100 / large ViTs (decision D-010; ViT-L/16 selected 2026-09-28). Separate image so the STL-10 functions
# above keep their exact original environment.
# ---------------------------------------------------------------------------
hf_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1", "pyyaml==6.0.2", "numpy==2.1.3",
        "datasets==5.0.0",
    )
    .add_local_dir("src", f"{PROJECT}/src")
    .add_local_dir("configs", f"{PROJECT}/configs")
)
IMAGENET100_ROOT = f"{VOLUME_ROOT}/data/imagenet-100"
HF_CACHE = f"{VOLUME_ROOT}/hf_cache"
DEFAULT_PROFILE_CONFIG = "imagenet100_vitl16_l2_modal_profile.yaml"


def _train_with_overrides(config_path: str, output_dir: str, data_overrides: dict | None = None) -> dict:
    """Run src.ijepa.train on a copy of `config_path` with output_dir (and optionally data keys) replaced."""
    import sys

    import yaml

    sys.path.insert(0, PROJECT)
    from src.ijepa.train import train

    with open(config_path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["training"]["output_dir"] = output_dir
    cfg["data"].update(data_overrides or {})
    patched = f"/tmp/{Path(output_dir).name}.yaml"
    with open(patched, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle)
    return train(patched, commit_callback=volume.commit)


SMOKE_CONFIG = f"{PROJECT}/configs/smoke_vitl16_modal.yaml"


def _smoke(gpu_tag: str) -> dict:
    """10 synthetic ViT-L/16 steps: validates hf_image, GPU forward/backward, checkpoint + commit."""
    import json
    import time

    import torch

    start = time.perf_counter()
    output_dir = f"{VOLUME_ROOT}/runs/smoke_vitl16_{gpu_tag}"
    result = _train_with_overrides(SMOKE_CONFIG, output_dir)
    with open(f"{output_dir}/metrics.jsonl", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {
        "gpu_tag": gpu_tag,
        "gpu": torch.cuda.get_device_name(),
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
        "steps": result["state"]["global_step"],
        "loss_first_last": [round(rows[0]["loss"], 4), round(rows[-1]["loss"], 4)],
        "peak_memory_gb": round(rows[-1]["peak_memory_gb"], 2),
        "images_per_second_cumulative": round(rows[-1]["images_per_second"], 1),
        "wall_seconds_in_container": round(time.perf_counter() - start, 1),
    }


@app.function(image=hf_image, gpu="L4", volumes={VOLUME_ROOT: volume}, timeout=60 * 30)
def smoke_vitl_l4():
    return _smoke("l4")


@app.function(image=hf_image, gpu="A10", volumes={VOLUME_ROOT: volume}, timeout=60 * 30)
def smoke_vitl_a10():
    return _smoke("a10")


@app.function(image=hf_image, gpu="L40S", volumes={VOLUME_ROOT: volume}, timeout=60 * 30)
def smoke_vitl_l40s():
    return _smoke("l40s")


@app.function(image=hf_image, gpu="A100-40GB", volumes={VOLUME_ROOT: volume}, timeout=60 * 30)
def smoke_vitl_a100_40gb():
    return _smoke("a100_40gb")


@app.function(image=hf_image, gpu="A100-80GB", volumes={VOLUME_ROOT: volume}, timeout=60 * 30)
def smoke_vitl_a100_80gb():
    return _smoke("a100_80gb")


@app.function(image=hf_image, gpu="H100", volumes={VOLUME_ROOT: volume}, timeout=60 * 30)
def smoke_vitl_h100():
    return _smoke("h100")


@app.function(image=hf_image, gpu="H200", volumes={VOLUME_ROOT: volume}, timeout=60 * 30)
def smoke_vitl_h200():
    return _smoke("h200")


SMOKE_FUNCTIONS = {
    "l4": smoke_vitl_l4,
    "a10": smoke_vitl_a10,
    "l40s": smoke_vitl_l40s,
    "a100_40gb": smoke_vitl_a100_40gb,
    "a100_80gb": smoke_vitl_a100_80gb,
    "h100": smoke_vitl_h100,
    "h200": smoke_vitl_h200,
}


@app.function(image=hf_image, volumes={VOLUME_ROOT: volume}, cpu=8, memory=16384, timeout=60 * 120)
def prepare_imagenet100():
    """CPU-only: download clane9/imagenet-100 into the Volume and build the Arrow cache once."""
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    snapshot_download("clane9/imagenet-100", repo_type="dataset", local_dir=IMAGENET100_ROOT, max_workers=8)
    volume.commit()
    counts = {}
    for split in ("train", "validation"):
        counts[split] = len(load_dataset("parquet", data_dir=f"{IMAGENET100_ROOT}/data", split=split, cache_dir=HF_CACHE))
    volume.commit()
    return {"root": IMAGENET100_ROOT, "cache": HF_CACHE, "rows": counts}


def _profile(config_name: str, gpu_tag: str) -> dict:
    import torch

    stem = Path(config_name).stem
    result = _train_with_overrides(f"{PROJECT}/configs/{config_name}", f"{VOLUME_ROOT}/runs/{stem}_{gpu_tag}")
    return {**result, "gpu": torch.cuda.get_device_name(), "memory_gb": torch.cuda.get_device_properties(0).total_memory / 2**30}


@app.function(image=hf_image, gpu="L40S", volumes={VOLUME_ROOT: volume}, cpu=8, memory=32768, timeout=60 * 30)
def profile_imagenet100_l40s(config_name: str = DEFAULT_PROFILE_CONFIG):
    return _profile(config_name, "l40s")


@app.function(image=hf_image, gpu="A100-80GB", volumes={VOLUME_ROOT: volume}, cpu=8, memory=32768, timeout=60 * 30)
def profile_imagenet100_a100(config_name: str = DEFAULT_PROFILE_CONFIG):
    return _profile(config_name, "a100_80gb")


@app.function(image=hf_image, gpu="H100", volumes={VOLUME_ROOT: volume}, cpu=8, memory=32768, timeout=60 * 30)
def profile_imagenet100_h100(config_name: str = DEFAULT_PROFILE_CONFIG):
    return _profile(config_name, "h100")


# ---------------------------------------------------------------------------
# RCDM-style latent-diffusion visualizer (decision D-011).
# ---------------------------------------------------------------------------
# Built from scratch: Modal requires add_local_* to be the last image steps.
rcdm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1", "pyyaml==6.0.2", "numpy==2.1.3",
        "datasets==5.0.0", "diffusers==0.35.2",
    )
    .add_local_dir("src", f"{PROJECT}/src")
    .add_local_dir("configs", f"{PROJECT}/configs")
)
VAE_CACHE = f"{VOLUME_ROOT}/hf_models"
RCDM_CONFIG = f"{PROJECT}/configs/rcdm_imagenet100_vitl16.yaml"
RCDM_CACHE_DIR = f"{VOLUME_ROOT}/rcdm_cache"


@app.function(image=rcdm_image, gpu="L4", volumes={VOLUME_ROOT: volume}, cpu=8, memory=12288, timeout=60 * 60 * 4)
def rcdm_cache_imagenet100(encoder_path: str, limit: int = 0, tag: str = "imagenet100_vitl16"):
    """Latents + pooled target-encoder features for ImageNet-100 train and validation.

    `limit` > 0 caches only the first N images of each split (pipeline checks with an early
    ViT-L checkpoint); the default 0 caches everything.
    """
    import sys

    sys.path.insert(0, PROJECT)
    from src.ijepa.rcdm_cache import build_cache

    results = {}
    for split, name in (("test", "validation"), ("train", "train")):  # "test" -> validation split
        results[split] = build_cache(
            encoder_path, split, f"{RCDM_CACHE_DIR}/{tag}_{name}.pt",
            data_root=f"{IMAGENET100_ROOT}/data", limit=limit or None,
            batch_size=128, num_workers=8, vae_cache=VAE_CACHE,
        )
        volume.commit()
    return results


def _rcdm_run(config_path: str, overrides: dict | None = None, cache_tag: str | None = None) -> dict:
    import sys

    import torch
    import yaml

    sys.path.insert(0, PROJECT)
    from src.ijepa.train_rcdm import train_rcdm

    with open(config_path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if cache_tag:
        cfg["cache"] = {"train": f"{RCDM_CACHE_DIR}/{cache_tag}_train.pt",
                        "val": f"{RCDM_CACHE_DIR}/{cache_tag}_validation.pt"}
    for key, value in (overrides or {}).items():
        cfg["training"][key] = value
    patched = f"/tmp/{Path(cfg['training']['output_dir']).name}.yaml"
    with open(patched, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle)
    result = train_rcdm(patched, commit_callback=volume.commit)
    return {**result, "gpu": torch.cuda.get_device_name()}


@app.function(image=rcdm_image, gpu="H100", volumes={VOLUME_ROOT: volume}, timeout=60 * 30)
def rcdm_profile_h100(cache_tag: str = "imagenet100_vitl16"):
    """300 steps of the ImageNet-100 RCDM config: throughput and memory before a long run."""
    return _rcdm_run(RCDM_CONFIG, {
        "total_steps": 300, "log_every": 50, "checkpoint_every": 300, "archive_every": 0,
        "sample_every": 300, "sample_steps": 25, "resume": False,
        "output_dir": f"{VOLUME_ROOT}/runs/rcdm_imagenet100_profile_h100",
    }, cache_tag=cache_tag)


@app.function(image=rcdm_image, gpu="H100", volumes={VOLUME_ROOT: volume}, timeout=60 * 60 * 24,
              retries=modal.Retries(max_retries=5, initial_delay=0.0))
def rcdm_train_imagenet100_h100():
    return _rcdm_run(RCDM_CONFIG)


@app.function(image=rcdm_image, gpu="L40S", volumes={VOLUME_ROOT: volume}, cpu=2, memory=12288,
              timeout=60 * 60 * 24, retries=modal.Retries(max_retries=5, initial_delay=0.0))
def rcdm_train_imagenet100_l40s():
    """Full RCDM run on an L40S (UNet peak 3.67 GB on the H100 profile). Retries resume from latest.pt."""
    return _rcdm_run(RCDM_CONFIG)


@app.function(image=hf_image, gpu="H100", volumes={VOLUME_ROOT: volume}, cpu=32, memory=65536, timeout=60 * 30)
def profile_imagenet100_h100_cpu32(config_name: str = DEFAULT_PROFILE_CONFIG):
    """Same as the H100 profile but with 32 reserved CPUs and 28 data workers (data-loader bottleneck test)."""
    import torch

    stem = Path(config_name).stem
    result = _train_with_overrides(f"{PROJECT}/configs/{config_name}", f"{VOLUME_ROOT}/runs/{stem}_h100_cpu32",
                                   data_overrides={"num_workers": 28})
    return {**result, "gpu": torch.cuda.get_device_name(), "cpus_reserved": 32, "num_workers": 28}


@app.function(image=hf_image, gpu="H100", volumes={VOLUME_ROOT: volume}, cpu=8, memory=32768,
              timeout=60 * 60 * 24, retries=modal.Retries(max_retries=5, initial_delay=0.0))
def pretrain_vitl_imagenet100_h100():
    """Main ViT-L/16 I-JEPA pretraining on ImageNet-100 (98 epochs). Retries resume from latest.pt."""
    import sys

    import torch

    sys.path.insert(0, PROJECT)
    from src.ijepa.train import train

    result = train(f"{PROJECT}/configs/imagenet100_vitl16_l2_modal.yaml", commit_callback=volume.commit)
    return {**result, "gpu": torch.cuda.get_device_name()}


@app.function(image=hf_image, gpu="A10", volumes={VOLUME_ROOT: volume}, cpu=8, memory=12288, timeout=60 * 60)
def profile_imagenet100_a10(config_name: str = "imagenet100_vitl16_l2_modal_profile_a10_accum4.yaml"):
    """A10 profile (12 GiB system RAM, the measured need is ~6 GB)."""
    return _profile(config_name, "a10")


@app.function(image=hf_image, gpu="A100-40GB", volumes={VOLUME_ROOT: volume}, cpu=8, memory=12288,
              timeout=60 * 60 * 8)  # no retries: a retry would restart the max_hours budget clock
def pretrain_vitl_b512_a100():
    """ViT-L/16, effective batch 512, 150-epoch schedule, stopped by training.max_hours (budget)."""
    import sys

    import torch

    sys.path.insert(0, PROJECT)
    from src.ijepa.train import train

    result = train(f"{PROJECT}/configs/imagenet100_vitl16_l2_a100_b512_150ep.yaml", commit_callback=volume.commit)
    return {**result, "gpu": torch.cuda.get_device_name()}


@app.function(image=hf_image, gpu="H100", volumes={VOLUME_ROOT: volume}, cpu=8, memory=12288,
              timeout=60 * 60 * 24, retries=modal.Retries(max_retries=2, initial_delay=0.0))
def pretrain_vitl_b512_h100(config_name: str = "imagenet100_vitl16_l2_h100_b512_150ep.yaml"):
    """ViT-L/16, effective batch 512 (2 x 256), 150-epoch schedule; ends at the account's Modal budget limit.
    Retries resume from latest.pt (the budget limit caps spend). `config_name` is a file in configs/;
    the `_ckpt10` variant is the same run with full + weights snapshots at epochs 110-140 every 10, every
    epoch 141-149, and epoch_150 at the end."""
    import sys

    import torch

    sys.path.insert(0, PROJECT)
    from src.ijepa.train import train

    result = train(f"{PROJECT}/configs/{config_name}", commit_callback=volume.commit)
    return {**result, "gpu": torch.cuda.get_device_name(), "config": config_name}


VITL_DECODER_PROFILE_CONFIG = "imagenet100_vitl16_decoder_predicted_profile.yaml"
VITL_DECODER_CONFIG = "imagenet100_vitl16_decoder_predicted_v1.yaml"


def _decoder_profile(config_name: str, gpu_tag: str) -> dict:
    """Short decoder run on the frozen ViT-L (config from configs/); returns throughput, memory and held-out rows."""
    import json
    import sys
    import time

    import torch
    import yaml

    sys.path.insert(0, PROJECT)
    from src.ijepa.train_decoder import train_decoder

    with open(f"{PROJECT}/configs/{config_name}", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    output_dir = f"{VOLUME_ROOT}/runs/{Path(config_name).stem}_{gpu_tag}"
    cfg["training"]["output_dir"] = output_dir
    patched = f"/tmp/{Path(output_dir).name}.yaml"
    with open(patched, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle)
    start = time.perf_counter()
    result = train_decoder(patched, commit_callback=volume.commit)
    rows = [json.loads(line) for line in open(f"{output_dir}/metrics.jsonl", encoding="utf-8")]
    train_rows = [r for r in rows if "masked_l1" in r]
    return {"gpu": torch.cuda.get_device_name(), "gpu_tag": gpu_tag, "config": config_name,
            "seconds": round(time.perf_counter() - start, 1), "last_train_row": train_rows[-1] if train_rows else None,
            "train_rows": train_rows[-4:], "holdout_rows": [r for r in rows if "val_masked_l1" in r][-2:],
            "state": result["state"]}


@app.function(image=hf_image, gpu="L4", volumes={VOLUME_ROOT: volume}, cpu=8, memory=12288, timeout=60 * 30)
def profile_decoder_vitl_l4(config_name: str = VITL_DECODER_PROFILE_CONFIG):
    return _decoder_profile(config_name, "l4")


@app.function(image=hf_image, gpu="A10G", volumes={VOLUME_ROOT: volume}, cpu=8, memory=12288, timeout=60 * 30)
def profile_decoder_vitl_a10g(config_name: str = VITL_DECODER_PROFILE_CONFIG):
    return _decoder_profile(config_name, "a10g")


@app.function(image=hf_image, gpu="L40S", volumes={VOLUME_ROOT: volume}, cpu=8, memory=12288, timeout=60 * 30)
def profile_decoder_vitl_l40s(config_name: str = VITL_DECODER_PROFILE_CONFIG):
    return _decoder_profile(config_name, "l40s")


@app.function(image=hf_image, gpu="H100", volumes={VOLUME_ROOT: volume}, cpu=8, memory=12288, timeout=60 * 30)
def profile_decoder_vitl_h100(config_name: str = VITL_DECODER_PROFILE_CONFIG):
    return _decoder_profile(config_name, "h100")


DECODER_PROFILE_FUNCTIONS = {"l4": profile_decoder_vitl_l4, "a10g": profile_decoder_vitl_a10g,
                             "l40s": profile_decoder_vitl_l40s, "h100": profile_decoder_vitl_h100}


@app.function(image=hf_image, gpu="L40S", volumes={VOLUME_ROOT: volume}, cpu=8, memory=12288,
              timeout=60 * 60 * 24, retries=modal.Retries(max_retries=2, initial_delay=0.0))
def train_decoder_vitl_l40s(config_name: str = VITL_DECODER_CONFIG):
    """Full ImageNet-100 ViT-L decoder training (resumes latest.pt; best.pt = best held-out masked L1).
    GPU choice is provisional until the decoder profiles have measured cost per step."""
    import sys

    import torch

    sys.path.insert(0, PROJECT)
    from src.ijepa.train_decoder import train_decoder

    result = train_decoder(f"{PROJECT}/configs/{config_name}", commit_callback=volume.commit)
    return {**result, "gpu": torch.cuda.get_device_name(), "config": config_name}


@app.local_entrypoint()
def main(action: str = "smoke", config: str = "", encoder: str = "", limit: int = 0, cache_tag: str = "", gpus: str = ""):
    if action == "prepare":
        print(prepare_stl10.remote())
    elif action == "smoke":
        print(smoke_gpu.remote())
    elif action == "train":
        call = pretrain_stl10.spawn()
        print({"function_call_id": call.object_id, "status": "submitted"})
    elif action == "continue-l4":
        call = continue_stl10_l4.spawn()
        print({"function_call_id": call.object_id, "status": "submitted"})
    elif action == "profile":
        print(profile_stl10.remote())
    elif action == "profile-l4":
        print(profile_stl10_l4.remote())
    elif action == "profile-l4-b256":
        print(profile_stl10_l4_b256.remote())
    elif action == "profile-t4":
        print(profile_stl10_t4.remote())
    elif action == "profile-decoder-l4":
        print(profile_masked_decoder_l4.remote())
    elif action == "train-decoder-l4":
        call = train_masked_decoder_l4.spawn()
        print({"function_call_id": call.object_id, "status": "submitted"})
    elif action == "adapt-decoder-final-l4":
        call = adapt_masked_decoder_final_encoder_l4.spawn()
        print({"function_call_id": call.object_id, "status": "submitted"})
    elif action == "evaluate":
        print(evaluate_stl10.remote())
    elif action == "evaluate-decoder":
        print(evaluate_decoder_controlled_l4.remote())
    elif action == "smoke-vitl":
        print(smoke_vitl_l4.remote())
    elif action == "smoke-vitl-all":
        import json

        calls = {tag: fn.spawn() for tag, fn in SMOKE_FUNCTIONS.items()}  # all GPUs in parallel
        for tag, call in calls.items():
            try:
                print(json.dumps(call.get()), flush=True)
            except Exception as error:  # report per-GPU failures instead of aborting the rest
                print(json.dumps({"gpu_tag": tag, "error": f"{type(error).__name__}: {error}"}), flush=True)
    elif action == "rcdm-cache-imagenet100":
        if not encoder:
            raise ValueError("--encoder </vol/... ViT-L checkpoint> is required")
        print(rcdm_cache_imagenet100.remote(encoder, limit, cache_tag or "imagenet100_vitl16"))
    elif action == "rcdm-profile-h100":
        print(rcdm_profile_h100.remote(cache_tag or "imagenet100_vitl16"))
    elif action == "rcdm-train-imagenet100":
        call = rcdm_train_imagenet100_h100.spawn()
        print({"function_call_id": call.object_id, "status": "submitted"})
    elif action == "rcdm-train-imagenet100-l40s":
        call = rcdm_train_imagenet100_l40s.spawn()
        print({"function_call_id": call.object_id, "status": "submitted"})
    elif action == "pretrain-vitl-imagenet100":
        call = pretrain_vitl_imagenet100_h100.spawn()
        print({"function_call_id": call.object_id, "status": "submitted"})
    elif action == "pretrain-vitl-b512-a100":
        call = pretrain_vitl_b512_a100.spawn()
        print({"function_call_id": call.object_id, "status": "submitted"})
    elif action == "pretrain-vitl-b512-h100":
        call = pretrain_vitl_b512_h100.spawn(config or "imagenet100_vitl16_l2_h100_b512_150ep.yaml")
        print({"function_call_id": call.object_id, "status": "submitted", "config": config or "default"})
    elif action == "prepare-imagenet100":
        print(prepare_imagenet100.remote())
    elif action == "profile-l40s-imagenet100":
        print(profile_imagenet100_l40s.remote(config or DEFAULT_PROFILE_CONFIG))
    elif action == "profile-a100-imagenet100":
        print(profile_imagenet100_a100.remote(config or DEFAULT_PROFILE_CONFIG))
    elif action == "profile-h100-cpu32-imagenet100":
        print(profile_imagenet100_h100_cpu32.remote(config or DEFAULT_PROFILE_CONFIG))
    elif action == "profile-a10-imagenet100":
        print(profile_imagenet100_a10.remote(config or "imagenet100_vitl16_l2_modal_profile_a10_accum4.yaml"))
    elif action == "profile-h100-imagenet100":
        print(profile_imagenet100_h100.remote(config or DEFAULT_PROFILE_CONFIG))
    elif action == "profile-decoder-vitl":
        import json

        tags = [t for t in (gpus or "l4,a10g,l40s,h100").split(",") if t]
        calls = {tag: DECODER_PROFILE_FUNCTIONS[tag].spawn(config or VITL_DECODER_PROFILE_CONFIG) for tag in tags}
        for tag, call in calls.items():  # all requested GPUs in parallel
            try:
                print(json.dumps(call.get()), flush=True)
            except Exception as error:
                print(json.dumps({"gpu_tag": tag, "error": f"{type(error).__name__}: {error}"}), flush=True)
    elif action == "train-decoder-vitl-l40s":
        call = train_decoder_vitl_l40s.spawn(config or VITL_DECODER_CONFIG)
        print({"function_call_id": call.object_id, "status": "submitted", "config": config or VITL_DECODER_CONFIG})
    else:
        raise ValueError(
            "action must be one of: prepare, smoke, profile, profile-l4, "
            "profile-l4-b256, profile-t4, profile-decoder-l4, train, "
            "continue-l4, train-decoder-l4, adapt-decoder-final-l4, evaluate, "
            "evaluate-decoder, smoke-vitl, smoke-vitl-all, rcdm-cache-imagenet100 (--encoder, --limit, --cache-tag), rcdm-profile-h100, rcdm-train-imagenet100, prepare-imagenet100, profile-l40s-imagenet100, "
            "profile-a100-imagenet100, profile-h100-imagenet100 (optional --config <file in configs/>)"
        )
