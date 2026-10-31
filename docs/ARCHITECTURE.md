# Architecture and implementation reference

This document describes what the code in `src/ijepa/` actually does. It was written 2026-09-28.

Updated 2026-09-29 with:
- the ViT-L/16 ImageNet-100 configuration;
- the opt-in training options (§4.5, §4.7);
- decoder conditioning modes (§4.8);
- the RCDM-style visualizer (§4.10).

Updated 2026-10-02 with:
- decoder model selection (held-out slice, `best.pt`, early stopping);
- the completion CLI (§4.9b);
- the RCDM EMA warmup.

Updated 2026-10-04 with the ViT size reference (§5).

Source tags:

| Tag | Meaning |
| --- | --- |
| `paper.md:N` | Line N of the paper extract |
| **[W]** | `work.md` |
| **[C]** | Computed from the repo's code and configs on 2026-09-28 |
| **[C2]** | Computed from the code and configs on 2026-09-29 (parameter counts on the meta device) |

Numbers without a tag are read directly from the code or configs cited alongside them.

> **Provenance references.** `paper.md` (a text extract of arXiv:2301.08243v3), `work.md`, `session.md`, `task.md`, `docs/EXPERIMENTS.md` and `docs/IMAGENET100_RUNBOOK.md` are the project's working records. They are not part of the public repository; references to them are kept to show where each number came from.

## 1. Three separate specifications

Keep these apart in configs, experiment names, and claims [W]:

| Specification | Representation objective | Decoder role |
| --- | --- | --- |
| I-JEPA paper | Average patch-level L2 in embedding space (`paper.md:275`) | Separate RCDM diffusion model, used only to visualize average-pooled representations (`paper.md:1151-1168`) |
| Official released code (archived 2024) [W] | LayerNorm on target features + Smooth L1 [W] | None |
| **This project** | LayerNorm'ed targets. **Smooth L1** in every STL-10 run; **L2** in the ImageNet-100 ViT-L run (from 2026-09-28) | Two parts: a deterministic spatial decoder from *predicted* target tokens to RGB patches (a project extension), and an RCDM-style latent-diffusion visualizer on pooled features (D-011; trained 2026-10-02) |

## 2. Data flow

```
                     ┌──────────────── pretraining (src/ijepa/train.py) ────────────────┐
image [B,3,224,224] ─┤                                                                   │
  (ImageNet-norm.)   │  full image ──► target encoder (EMA, no_grad) ──► LayerNorm(D) ──► gather target idx ──► s_y  [4,B,Nt,384]
                     │                                                                                           │
                     │  MultiBlockMaskSampler ──► context idx [B,Nc], target idx [4,B,Nt]                        │ smooth L1
                     │                                                                                           │
                     │  context idx ──► context encoder ──► s_x [B,Nc,384] ──► predictor(s_x, ctx idx, tgt idx) ──► ŝ_y [4,B,Nt,384]
                     │                                                                                           │
                     │  AdamW on context encoder + predictor ──► then EMA: θ̄ ← m·θ̄ + (1−m)·θ                    │
                     └───────────────────────────────────────────────────────────────────────────────────────────┘

                     ┌──────────────── decoder training (src/ijepa/train_decoder.py) ───┐
                     │  frozen I-JEPA ──► ŝ_y [4,B,Nt,384] ──► MaskedPatchDecoder ──► RGB patches [4,B,Nt,768]
                     │  L1 vs the true target patches (ImageNet-normalized pixel space)
                     │  compose: average overlapping blocks, copy visible patches bit-for-bit ──► completed image
                     └───────────────────────────────────────────────────────────────────┘

                     ┌──────────────── RCDM visualizer (rcdm_cache.py → train_rcdm.py) ─┐
                     │  cache once: frozen SD-VAE(image) ──► latent z [4,28,28];  frozen target encoder ──► mean-pool ──► h [D]
                     │  train: z_t = DDPM.add_noise(z, ε, t);  UNet(z_t, t, MLP(h)) ──► ε̂;  loss = MSE(ε̂, ε)
                     │  sample: DDIM from noise, conditioned on h ──► z ──► SD-VAE decode ──► image
                     └───────────────────────────────────────────────────────────────────┘
```

The diagram shows ViT-S dimensions (D = 384); for ViT-L, D = 1024.

With `data.masks_in_loader: true`, the DataLoader workers produce the masks (§4.1). With `training.grad_accum_steps: k`, k micro-batches are accumulated per optimizer and EMA step (§4.5).

## 3. Tensor contracts (ViT-S/16 at 224 px)

- **Grid.** 14×14 = 196 patches of 16×16×3 = 768 values each.
- **`MaskBatch.context`** is `[B, Nc]`: sorted, unique patch indices, disjoint from every target index of the same image (tested).
- **`MaskBatch.targets`** is `[M=4, B, Nt]`. All four blocks share one `h×w` shape per batch, so `Nt = h·w`. Blocks may overlap each other.
- **Measured distributions** over 2,000 batches with the real config [C]:
  - `Nt` is 25–42 tokens (median 35).
  - The target union per image covers 41–150 patches, mean 93.7, i.e. **47.8% of the image**.
  - `Nc` depends on batch size (section 4.1).
- **Encoder outputs.**
  - Context encoder: `[B, Nc, 384]`.
  - Target encoder: `[B, 196, 384]` before gathering; targets and predictions are both `[4, B, Nt, 384]`.
  - `latent_loss` raises if the prediction and target shapes differ.
- **Decoder.** Output `[4, B, Nt, 768]`. `compose_masked_images` returns masked and completed images `[B, 3, 224, 224]` plus a per-patch target mask `[B, 196]`.

## 4. Components

### 4.1 Multi-block masks (`masks.py`)

**Block shape.**
- `area = max(1, round(196·U(scale)))`, `ratio = U(aspect)`.
- `h = round(√(area·ratio))`, `w = round(√(area/ratio))`, each clamped to [1, 14].
- Targets use scale (0.15, 0.20) and aspect (0.75, 1.5). The context uses scale (0.85, 1.0) and aspect 1.0. This matches `paper.md:960-962`.

**Per batch.**
- One target shape and one context shape are drawn per batch; locations are drawn independently per image.
- The context is the context block minus the union of the 4 targets.
- Sampling retries up to 20 times to reach `min_context_patches` (10), then raises.

**Batch-minimum trimming.**
- Each image's context is randomly subsampled to the **batch minimum** length so contexts can be stacked. This is the implementation of the paper's "identical context-mask size on a co-located GPU" constraint (`paper.md:963-965`).
- Consequence: **the context length shrinks as the batch grows** [C]:

| Batch | Mean context tokens | Mean fraction visible |
| ---: | ---: | ---: |
| 1 | 89.9 | 45.9% |
| 8 | 72.0 | 36.7% |
| 64 (laptop) | 60.5 | 30.9% |
| 128 (Modal, decoder eval) | 58.1 | 29.6% |

**RNG.** Masks come from a dedicated `torch.Generator` seeded `seed + global_step` when a run starts or resumes. The generator state is not checkpointed.

**Loader-side masks (`MaskCollator`, opt-in via `data.masks_in_loader: true`).**
- The collate function stacks images and labels and calls the same `sampler.sample(B, generator)` inside each DataLoader worker. This mirrors the official design (`paper.md:965-968`).
- Each worker lazily creates its own generator from `torch.initial_seed() % 2**63`. With this option the main-process `seed + global_step` generator is not used.
- Batches become `(images, labels, MaskBatch)`, and the training loop reads `batch[2]`. The `MaskBatch` contract is identical (tested by `test_mask_collator_matches_sampler_contract`).
- Motivation: mask sampling took 69.2 ms per batch-128 step on the laptop CPU, against a 252 ms H100 step [W].

### 4.2 Vision Transformer (`model.py: VisionTransformer`)

- `Conv2d(3, D, 16, stride 16)` patch embedding, then a fixed 2-D sin-cos positional embedding (a persistent buffer).
- When `indices` is given, tokens are gathered **after** the positional embedding is added. This is how the context encoder sees only context patches.
- Pre-norm blocks: `x + MHA(LN(x))`, then `x + MLP(LN(x))` with GELU and MLP ratio 4. `nn.MultiheadAttention` is used with no dropout and no drop-path.
- A final LayerNorm; no `[cls]` token, as in the paper (`paper.md:951-952`).
- Linear layers are initialized with `trunc_normal(std=0.02)` and zero bias.

### 4.3 Predictor (`model.py: Predictor`)

1. `Linear(384→192)` on the context tokens, plus the predictor's own sin-cos positional embedding.
2. Repeat the result for each of the M targets.
3. Append `Nt` copies of a learned `mask_token` plus the target positions.
4. Run 6 pre-norm blocks (6 heads) over the `M·B` sequences.
5. Take the last `Nt` outputs, apply LayerNorm, then `Linear(192→384)`.

Each target block is predicted in its own sequence. The steps above are for ViT-S/16. The ViT-L/16 config uses `Linear(1024→384)`, 12 blocks with 16 heads, and `Linear(384→1024)`. That matches the paper's predictor for ViT-L (`paper.md:948-951`).

### 4.4 Targets and loss

- `IJEPAModel.targets` runs the target encoder on the **full image** under `no_grad`.
- With `normalize_targets: true` (every config), it applies `F.layer_norm` over the feature dimension (no affine) and only then gathers each target block.
- `smooth_l1` is `F.smooth_l1_loss` with the defaults (β = 1, mean reduction).
- `l2` is `(pred − target)²` averaged over *all elements*. The paper sums over the patches of each block and averages over the M blocks (`paper.md:275`), so the two differ by a constant factor and hence in effective learning rate. The ImageNet-100 ViT-L run uses `l2`.

### 4.5 Optimization (`train.py`)

**Optimizer.**
- AdamW with PyTorch default betas and eps, covering **only** the context encoder and predictor. The target encoder has `requires_grad_(False)` and is absent from the optimizer (tested).
- Two parameter groups: no weight decay for 1-D parameters and biases, weight decay for everything else. The predictor's `mask_token` has shape `[1,1,192]`, so it *is* weight-decayed.

**Schedules**, all indexed by `global_step` and `total_steps`, and therefore continuous across resumes:

| Quantity | Schedule |
| --- | --- |
| LR | Linear warmup `start_lr → lr` over `warmup_steps`, then cosine to `final_lr` |
| Weight decay | Linear `weight_decay → final_weight_decay` (0.04 → 0.4), applied only to the decay group |
| EMA momentum | Linear `ema[0] → ema[1]` (0.996 → 1.0) |

**Step order.**
1. Autocast forward, then loss. A non-finite loss raises `FloatingPointError`.
2. `scaler.scale(loss).backward()`, then `unscale_`, then `clip_grad_norm_(1.0)`.
3. `scaler.step`, then `scaler.update`.
4. **`ema_update(m)` after the optimizer step** (tested with a known scalar case).

**Precision.** `precision: bf16|fp16|fp32`. `GradScaler` is enabled only for fp16 (the T4 profile). TF32 matmul is enabled.

**Opt-in options added 2026-09-28.** The defaults reproduce the August behaviour: the smoke losses match to 4 decimals.

| Key | Effect |
| --- | --- |
| `training.grad_accum_steps: k` | Each micro-batch's loss is divided by `k` before `backward()`. Gradients are zeroed only at the first micro-batch. The optimizer step, `ema_update`, LR/WD/EMA schedules, and `global_step` advance once per `k` micro-batches. `steps_per_epoch = len(dataset) // batch_size // k`. Tested: two accumulated half-batches equal the full-batch gradient. |
| `training.sync_every_step: false` | The `isfinite(loss)` check, which forces a GPU→CPU sync, runs only on log and checkpoint steps. `feature_stats` runs only on log steps; the logged values are unchanged. |
| `data.masks_in_loader: true` | §4.1 (`MaskCollator`). |
| `training.max_hours: h` | After any optimizer step with `h` hours elapsed *since this process started*, the trainer saves `latest.pt` and `step_XXXXXXX.pt`, commits, and returns `{"stopped_early": True}`. `0` disables it. |
| `training.epoch_checkpoint_from: N` | §4.7. |
| `training.epoch_checkpoint_every: K` | Default 1. Only epochs N, N+K, N+2K, … are snapshotted (§4.7). Added 2026-09-29. |
| `training.epoch_checkpoint_full: true` | At each snapshot epoch, also writes an immutable full resumable `epoch_NNN.pt` (§4.7). Added 2026-09-29. |
| `training.epoch_checkpoint_dense_from: E` | Snapshot every epoch from E on, in addition to the every-K rule. Added 2026-09-29. |
| `training.epoch_checkpoint_final: true` | After the last step, snapshot tagged with the completed-epoch count (e.g. `epoch_150` at step 37,050), taken at the *end* of that epoch, whereas the other snapshots are taken at an epoch's first step. Added 2026-09-29. |

With accumulation, `batch_size` is the **micro-batch**. The ViT-L main run uses 256 × 2 = 512 effective, with LR 2.5e-4 (the paper's 1e-3 at batch 2048, scaled linearly).

### 4.6 Logged diagnostics (`metrics.jsonl`, every `log_every` steps)

| Field | Meaning |
| --- | --- |
| `loss`, `lr`, `wd`, `ema` | As the names say |
| `grad_norm` | Total norm *before* clipping |
| `feature_std` | For each image, mean the normalized target tokens over all target blocks to get a `[B,384]` matrix. Then take the per-dimension std across the batch, averaged over dimensions. Collapse would drive it toward 0. |
| `mean_cosine` | Mean off-diagonal cosine similarity between those per-image vectors. Collapse would drive it toward 1. |
| `images_per_second` | Images since *this process* started or resumed (`(step + 1 − start_step) × batch_size × grad_accum_steps`), divided by elapsed time. Includes startup and checkpoint saves. |
| `peak_memory_gb` | `torch.cuda.max_memory_allocated()` in GiB, a running maximum |

### 4.7 Checkpoints and resume (`train.save_checkpoint` / `load_checkpoint`, shared by both trainers)

- **Payload:** `{model, optimizer, scaler, state{global_step, epoch, batch_in_epoch}, config, rng{python, numpy, torch, cuda}}`.
- **Atomic write:** save to `*.pt.tmp`, then `os.replace`.
- **Files:** `latest.pt` every `checkpoint_every` steps and at the end; immutable `step_XXXXXXX.pt` every `archive_every`; then `commit_callback()` (Modal `volume.commit`).
- **Resume:** when `resume: true` and `latest.pt` exists, the full state is restored. The data loader is rebuilt with shuffle seed `seed + epoch`, and batches before `batch_in_epoch` are skipped.
- **Batch-size change:** `reset_data_cursor_on_resume: true` zeroes `batch_in_epoch` so a checkpoint can continue at a new batch size (tested). The saved torch and CUDA RNG ByteTensors are moved to CPU before being restored. That fixed restoring a checkpoint loaded with `map_location="cuda"` [W].
- **Per-epoch snapshots** (`training.epoch_checkpoint_from: N`; CPU-tested):
  - From data epoch N on, the first optimizer step of each new epoch writes a rolling full `latest.pt` plus `epoch_NNN_weights.pt`, then commits.
  - The snapshot is `{model: state_dict in bf16 (non-float tensors unchanged), state, config, note}`. It has no optimizer, so it cannot resume, but `IJEPAModel.load_state_dict` accepts it (the copy casts bf16 to fp32).
  - ViT-L sizes: snapshot 1,257,768,988 bytes; `latest.pt` 5,117,152,475 bytes.
  - **Cadence and full copies (2026-09-29).**
    - With `epoch_checkpoint_every: K`, only epochs where `(epoch − N) % K == 0` are snapshotted.
    - With `epoch_checkpoint_full: true`, each snapshot epoch also gets `epoch_NNN.pt`, the full `save_checkpoint` payload: model, optimizer, scaler, state, config, RNG, ≈5.1 GB for ViT-L.
    - `epoch_checkpoint_dense_from: E` adds every epoch from E on.
    - `epoch_checkpoint_final: true` adds a snapshot after the last step, tagged with the completed-epoch count (`epoch_150` at step 37,050).
    - The ViT-L relaunch (`..._ckpt10.yaml`) uses from 60, every 10, full, dense from 141, and final.
  - **No overwrite (2026-09-29).** A file that already exists is skipped, and the save block runs only if a file is missing.
    - Before this change, a resume re-wrote the resumed epoch's snapshot one step later, because `last_epoch_saved` starts at `epoch − 1`: `epoch_075_weights.pt` on the second Modal account was written at step 18,527.
    - Now a resume inside a saved epoch writes only the missing file. For example, if the weights exist but the full copy does not, the full copy is written at the resumed step.
    - Tested by `test_epoch_snapshots_every_n_with_full_checkpoints_are_never_overwritten`.
  - `latest.pt` is written at the *first* step of an epoch, and still every `checkpoint_every` steps. A killed run resumes from the later of the two, at most one epoch back.
  - For the ViT-L budget stop, 5–14 updates were lost: `latest.pt` is at 18,526, the last logged step is 18,531, and the log interval is 10.

### 4.8 Decoder and composition (`decoder.py`, `train_decoder.py`)

**Loading.** `load_frozen_ijepa` rebuilds `IJEPAModel` from the checkpoint's embedded config, freezes it, and sets eval mode. The model geometry, `mask` config, and `data` config come from the **encoder checkpoint**; the decoder YAML overrides only data-loader fields.

**Forward pass.**
- `MaskedPatchDecoder` takes the predicted tokens `[M,B,Nt,384]`.
- It flattens them to `M·B` sequences, applies `Linear(384→256)` plus a sin-cos positional embedding for the target indices, runs 4 blocks (8 heads), then `LayerNorm → Linear(256→768)`.
- **Each target block is decoded independently.** The decoder never sees visible pixels, context-encoder tokens, or other blocks.

**Loss.** L1 between predicted and true target patches in **ImageNet-normalized** pixel space. MSE is logged but not optimized. There is no per-patch normalization.

**Composition.**
- `compose_masked_images` scatter-adds the block predictions, divides by the per-patch counts (overlaps are averaged), and uses `torch.where` so visible patches are the original tensor values.
- Predictions are cast to the image dtype first. That was the BF16 fix [W], and it is tested.
- Masked patches are zero in normalized space, which is why they appear as the mean grey-brown after denormalization.

**Warm start.** `decoder_init_checkpoint` loads decoder weights only; the optimizer, schedule, and step counter start fresh.

**Model selection (opt-in, 2026-10-02; tested by `test_decoder_holdout_eval_best_checkpoint_and_early_stop`).**
- `training.holdout_size`/`holdout_seed`: a deterministic slice of the **training** split (`train` for ImageNet-100, `unlabeled` for STL-10), re-opened with the eval transform and excluded from training. The dataset's validation split stays the untouched benchmark.
- `training.eval_every` (+ `eval_batch_size`, `eval_mask_seed` 20260831): held-out masked L1/MSE with the **same masks at every evaluation**.
- `best.pt` = the best held-out masked L1. `best_val_masked_l1` and `evals_since_best` are stored in the checkpoint state and survive resumes.
- `training.early_stop_patience`: stop after N evaluations without improvement.
- `data.masks_in_loader`: `MaskCollator`, as in pretraining.

**Conditioning** (`conditioning:` in the decoder YAML; `decoder_conditioning()`; tested by `test_decoder_conditioning_modes`):
- `predicted` (default) feeds `IJEPAModel.predictions()`, from the context encoder and predictor.
- `oracle` feeds `IJEPAModel.targets()`: the LayerNorm'ed target-encoder tokens of the full image, gathered at the target indices.

`oracle` is an upper-bound diagnostic only, because it sees the hidden region. On STL-10 it reaches a training-batch masked L1 of ≈0.14, against ≈0.50 for `predicted`, with visibly structured fills. The blur is therefore mainly a property of the predicted features (`docs/EXPERIMENTS.md` §2.6).

### 4.9 Evaluation

- **`evaluate.evaluate_stl10`** uses the **target encoder**.
  - Transforms: `Resize(256) → CenterCrop(224)` then ImageNet normalization.
  - Features are token-averaged, then L2-normalized.
  - Weighted k-NN with k = 20 and weights `exp(sim/0.07)`. The bank is the labeled `train` split and the queries are `test`.
- **`evaluate_decoder.evaluate_pairing`** computes elementwise masked L1/MSE (sums divided by element count) over a fixed subset with a fixed mask seed, under BF16 autocast. It saves the first batch as a grid.
  - Optional `conditioning` overrides the decoder's own mode.
  - `shuffle_latents=True` rolls the conditioning tokens by one image within the batch (`.roll(1, dims=1)`).
  - The local `main()` takes `--encoder`, a repeatable `--decoder NAME=PATH`, `--output-dir`, `--data-root`, `--examples 1024`, `--batch-size 128`, `--mask-seed 20260831`, `--num-workers`, and `--with-shuffled`.
- **`scripts/eval_vitl_snapshots.py`** runs the §4.9 k-NN protocol for ImageNet-100 on the laptop.
  - It loads `epoch_*_weights.pt` or full checkpoints and uses the target encoder under BF16 autocast.
  - Bank: `train` (126,689) with eval transforms. Queries: `test` → validation (5,000).
  - Data paths default to `data/imagenet-100/data` and `data/hf_cache` (`--data-root`, `--cache-dir`; changed from hard-coded local paths 2026-10-31). `--random-init` adds an untrained ViT-L of the same architecture.

### 4.9b Completion CLI (`complete.py`, 2026-10-02; 4 tests in `CompletionCliTests`)

- **Preprocessing:** identical to the training eval transform. A test compares it with `data._eval_transform`.
- **`crop_geometry` + `rect_to_patch_indices`:** a rectangle in original-image pixels is mapped through torchvision's integer Resize/CenterCrop rules. Every touched patch becomes one target block, and the context is all other patches (`--context-tokens N` optionally subsamples it, since training batches saw ≈60).
- **Random masks** use the encoder checkpoint's mask config, seeded.
- **`load_models` refuses:**
  - a decoder whose recorded `encoder_checkpoint` file name differs (`--allow-mismatch`);
  - a token-width mismatch;
  - `conditioning: oracle` (`--allow-oracle`).
- **Outputs:** `original/masked/completed/comparison.png` and `metadata.json` (geometry, patch counts, masked L1/MSE against the true crop, and `visible_pixels_exact`, which is checked on every run).
- **CUDA OOM** produces an error suggesting `--device cpu`.
- **RCDM mode** (`--rcdm RCDM.pt`, mutually exclusive with `--decoder`; 3 tests in `RCDMCompletionTests`):
  - `rcdm_condition` rebuilds the pooled feature RCDM was trained on from visible pixels only: target-encoder tokens of the visible patches, plus predictor tokens for the hidden patches mapped back from per-token LayerNorm space with the visible tokens' mean/std. `visible` uses the visible tokens alone; `oracle` (the full-image feature) needs `--allow-oracle`.
  - `rcdm.inpaint` generates only the latent cells touching the hole; the rest are reset every step to the noised VAE latents of the masked image (hidden pixels at the mean colour), so hidden pixels never reach the VAE. `--resample R` adds RePaint repeats.
  - A test checks that two images differing only inside the hole give bit-identical completions.
  - Caveat: RCDM was not trained for inpainting, so fills can be a shade off in tone and do not always continue edges across the seam.

### 4.10 RCDM-style visualizer (`rcdm.py`, `rcdm_cache.py`, `train_rcdm.py`; decision D-011)

The purpose follows the paper's visualization (`paper.md:1151-1168`): show what information the *average-pooled* representation keeps, by sampling several images per feature.

**Frozen parts.**
- SD-VAE `stabilityai/sd-vae-ft-mse` (MIT). Its latents are the posterior mean × `scaling_factor`, 4×28×28 for 224 px inputs.
- The I-JEPA target encoder, whose token mean is the feature `h`. This is the same feature k-NN uses.
- The pixel pipeline stays ImageNet-normalized and is mapped to the VAE's [−1, 1] range inside `encode_latents`.

**`RCDMUNet(feature_dim)`.**
- A diffusers `UNet2DModel` on the latents: `block_out_channels (128, 256, 512)`, 2 layers per block, attention at the two lower resolutions, head dim 64.
- `cond_proj = Linear(D→512) → SiLU → Linear(512→512)` produces a vector. It is added to the time embedding (`class_embed_type="identity"`), which then scale/shift-modulates every residual block (`resnet_time_scale_shift="scale_shift"`). This is the latent-space analogue of RCDM's conditioning.
- Parameters [C2]: UNet 66,968,452. The MLP has 787,456 for D = 1024 (ViT-L), giving **67,755,908** in total. For D = 384 (ViT-S) the MLP has 459,776, giving 67,428,228; that is the "67.4M" quoted in `session.md` §10.

**Training (`train_rcdm.py`).**
- DDPM with `scaled_linear` betas 0.00085–0.012 over 1,000 steps. Epsilon prediction, MSE loss.
- An EMA copy of the UNet (`ema_decay` 0.9999 in `configs/rcdm_imagenet100_vitl16.yaml`).
  - With `optimization.ema_warmup: true` (set in that config since 2026-10-02) the decay is `min(ema_decay, (1 + t) / (10 + t))`, reaching 0.9999 at t = 89,990.
  - Without it, the sample grids come from an EMA that is still mostly the random init early on: 0.9999^t of it, i.e. 0.97 at step 300 (the profile's noise grid) and 0.61 at step 5,000.
- Sample grids every `sample_every` steps: several DDIM samples (50 steps, η = 0) per fixed validation feature.
- The trainer reads only the cache built by `rcdm_cache.build_cache`, so neither the VAE nor the encoder runs during training.
- There is no held-out slice: batches are drawn with replacement from the whole training cache (`train_rcdm.py:108`), so every training image is seen (≈50 times in 100k steps). Only the validation split is unseen. Decoder v1, by contrast, holds out 5,000 training images (`train_decoder.py:184-193`).
- `training.device` can override the device.
- Batch 64 and 100k steps. Measured throughput: H100 ≈1,100 img/s (300-step profile), L40S 921 img/s (start of the full run). Peak GPU memory: 3.63 GB early in the full run, 3.93 GB by its end.

**Status.**
- 6 unit tests (`RCDMTests`), including a check that the conditioning changes the output and two for the EMA warmup.
- One CPU end-to-end run: cache, train, sample, checkpoint, resume.
- SD-VAE round trip: mean PSNR 31.5 dB on 8 STL-10 test images, on CPU [W].
- First full run on ViT-L features: 100k steps on Modal (L40S), 2026-10-02, $4.22. Samples keep object type, colour, texture and pose; faces and fine detail are distorted (`docs/EXPERIMENTS.md`).
- Evaluation (`scripts/eval_rcdm.py`, 1,000 validation images × 4 samples): every sample retrieves its own source top-1 among 5,000 validation features; samples score 51.8% weighted 20-NN class accuracy against 61.0% for the real images. These are measured with the conditioning encoder itself (self-consistency, not perceptual quality).

## 5. Model sizes [C]

| Module | Parameters |
| --- | ---: |
| Context encoder (ViT-S/16: dim 384, depth 12, 6 heads) | 21,589,632 (+75,264 positional-embedding buffer) |
| Target encoder (EMA copy) | 21,589,632 |
| Predictor (dim 192, depth 6, 6 heads) | 2,817,792 |
| **Trainable during pretraining** | **24,407,424** |
| Masked-patch decoder (dim 256, depth 4, 8 heads) | 3,455,488 |

**ViT-L/16** (`configs/imagenet100_vitl16_l2_h100_b512_150ep.yaml`) [C2]:

| Module | Parameters |
| --- | ---: |
| Context encoder (dim 1024, depth 24, 16 heads) | 303,098,880 (+ a 196×1024 positional-embedding buffer) |
| Target encoder (EMA copy) | 303,098,880 |
| Predictor (dim 384, depth 12, 16 heads) | 22,082,560 |
| **Trainable during pretraining** | **325,181,440** |
| Masked-patch decoder v1 (dim 256, depth 4, 8 heads; input 1024) | 3,619,328 |
| RCDM UNet + conditioning MLP (D = 1024) | 67,755,908 |

A checkpoint's `state_dict` holds slightly more numbers than these parameter counts: the fixed sin-cos positional embeddings are persistent buffers (196 × 1024 = 200,704 per encoder, 196 × 384 = 75,264 for the predictor). Summing the state dict of `epoch_150_weights.pt` gives 303,299,584 for the target encoder and 22,157,824 for the predictor (checked 2026-10-04).

**ViT size reference** (for encoders brought from elsewhere):

| Model | Embedding dim | Blocks | Heads | MLP hidden | Encoder parameters | Source |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ViT-Tiny/16 | 192 | 12 | 3 | 768 | ≈5.7M | standard (DeiT); not in this repo |
| ViT-Small/16 | 384 | 12 | 6 | 1536 | 21,589,632 | this repo's STL-10 encoder [C] |
| ViT-Base/16 | 768 | 12 | 12 | 3072 | ≈85.6M | this repo's ViT-B config, computed (`work.md`, 2026-09-28) |
| **ViT-Large/16** (main model) | **1024** | 24 | 16 | 4096 | **303,098,880** | this repo's checkpoint [C2] |

At 224 px with 16 px patches every size gives 196 tokens: per-patch embeddings are 196 × dim and the pooled embedding is one dim-sized vector.

A decoder only reads tokens of the width it was trained on (the CLI checks `input_proj.weight.shape[1]`), so each width needs its own decoder.

Measured memory for ViT-L training in BF16 [W]:

| Setting | Peak allocated |
| --- | ---: |
| Micro-batch 16 | 8.05 GiB |
| Micro-batch 128 | 23.4–26.0 GiB |
| Micro-batch 256 | 42.9 GiB in the profile; 50.38 GiB by the end of the main run, as rare large mask combinations occur |

The Modal dashboard shows the allocator's *reserved* memory (≈71 GB mid-run at micro-batch 256), not these allocated peaks.

## 6. Deviations from the paper

| Aspect | Paper (source) | This project |
| --- | --- | --- |
| Data | ImageNet-1K | **STL-10 phase:** `unlabeled`, 100k images at 96×96, upsampled by `RandomResizedCrop(224, scale 0.3–1.0)`, no flip. **ImageNet-100 phase:** `clane9/imagenet-100` (126,689 train images, 100 classes) at a 160 px short side, so its 224 crops are also upsampled. Same augmentation. |
| Backbone | ViT-B/16 … ViT-G/16 (`paper.md:946-953`) | ViT-S/16 (STL-10); **ViT-L/16** (ImageNet-100) |
| Predictor | Width fixed at 384, heads = backbone heads; depth 6 for ViT-B, 12 for L/H (`paper.md:948-951`) | ViT-S: width **192**, depth 6, 6 heads. ViT-L: width 384, depth 12, 16 heads, as in the paper. |
| Batch size | 2048 (`paper.md:954`) | 64 (laptop); 128 (Modal continuation); **512 effective** for ViT-L (2 × 256 accumulated) |
| LR | 1e-4 → 1e-3 over the first 15 epochs, then cosine to 1e-6 (`paper.md:955-956`) | STL-10: 1e-5 → 1.5e-4 over 10k steps (laptop), or → 3e-4 over 5k steps (Modal); cosine to 1e-6. ViT-L: 1e-5 → **2.5e-4** (1e-3 × 512/2048) over 1,852 steps (7.5 epochs, 5% of the schedule), then cosine to 1e-6. |
| Weight decay | Linear 0.04 → 0.4 (`paper.md:956`) | Same |
| EMA | 0.996 → 1.0 linear (`paper.md:958-959`) | Same |
| Masks | 4 targets (0.15–0.2, aspect 0.75–1.5); 1 context (0.85–1.0, unit aspect) minus targets; sizes equalized per GPU (`paper.md:960-965`) | Same ranges. One block *shape* per batch; context randomly subsampled to the batch-minimum length |
| Objective | L2 averaged over M targets (`paper.md:275`) | STL-10: Smooth L1 on LayerNorm'ed targets, following the released code [W]. ViT-L: L2 on LayerNorm'ed targets, averaged over all elements (§4.4). |
| Training length | 300-epoch ablations (`paper.md:1073, 1081`) | STL-10: 200k steps, i.e. 12.8M images (laptop) or 21.9M (Modal branch) [C]. ViT-L: a 150-epoch schedule of 37,050 updates (≈19.0M images), complete since 2026-09-29. |
| Evaluation | Linear probe (LARS, batch 16384, 50 epochs) and 1% low-shot (`paper.md:985-995`) | Weighted 20-NN: STL-10, and ImageNet-100 validation (final ViT-L 60.02%; random init 8.66%). A frozen linear probe on ImageNet-100 (final ViT-L 65.24% top-1 / 88.00% top-5; random init 16.96%). The paper's 1% low-shot protocol was not run, and there is no MAE baseline yet. |
| Visualization | RCDM diffusion on average-pooled representations, 300k iterations (`paper.md:1151-1168`) | A deterministic per-patch decoder on predicted spatial tokens, for masked completion. Plus an RCDM-style UNet in **SD-VAE latent space**, rather than pixels, conditioned on pooled features (§4.10); trained for 100k steps on 2026-10-02. |

## 7. Known limitations and code caveats

All of these were found by reading or running the code.

1. **The decoder has no context input** (section 4.8). Fills cannot blend with the surrounding pixels, which contributes to the block seams. `work.md`'s Track B design called for "predicted target tokens plus visible/context features", and only the first half is implemented.
   - Correction, 2026-09-28/29: the oracle diagnostic shows this is *not* the main cause of the blur. With oracle target tokens, the same context-free decoder produces structured fills (training-batch L1 ≈0.14 vs ≈0.50).
   - The quality of the predicted features is the larger limitation.
2. **Train-time and inference-time context lengths differ.** The batch-minimum trimming (section 4.1) gives about 60 context tokens at batch 64 but about 90 at batch 1. A single-image completion CLI would feed the predictor longer contexts than it was trained on. Whether that helps or hurts has not been measured.
3. **Resume is deterministic, but bit-identity is not guaranteed.**
   - Skipped batches are still loaded and augmented, so a mid-epoch resume spends time re-decoding.
   - The mask generator is reseeded rather than restored. With `masks_in_loader`, masks come from the workers' seeds.
   - Observed 2026-09-29: two ViT-L resumes on an H100, with `masks_in_loader`, reproduced the pre-stop row to every printed digit (steps 18,531 and 25,001). This is an observation, not a guarantee.
   - There is no test comparing "N steps" against "K steps + resume + N−K steps". The roadmap's verification plan asked for one.
4. **`evaluate_stl10()` is dataset-agnostic in behaviour but not in naming.**
   - The result key is `stl10_test_accuracy`, and `data_root` defaults to `/vol/data`.
   - For `imagenet100` it uses the full 126,689-image train split as the bank and the 5,000 validation images as queries.
   - `num_workers=8` is hard-coded.
   - Without `output_path` it writes `knn_eval.json` next to the checkpoint.
5. **`evaluate_decoder.evaluate_decoder_pairings()` hard-codes four `/vol/...` checkpoint paths**, so that function runs only on Modal. The local `main()` added 2026-09-28 takes explicit paths.
6. **`scripts/show_progress.py` works for encoder runs only.** It reads `loss`, `feature_std`, and `mean_cosine`, and its ETA assumes the run was never resumed.
7. **`scripts/compare_final_encoders.py` overwrites** `artifacts/knn_eval_{laptop,modal}_step_200000.json` and `artifacts/encoder_step200k_comparison.json`, whose hashes are recorded in `work.md`.
8. **Modal images.**
   - The original `image`, used by the STL-10 functions, installs no `datasets`.
   - The ImageNet-100 functions use `hf_image` (+ `datasets==5.0.0`), and the RCDM functions use `rcdm_image` (+ `datasets`, `diffusers==0.35.2`).
   - All three run Python 3.11 while the laptop runs 3.10.
   - Modal requires `add_local_dir` to be the last build step: a `pip_install` after it fails at App creation. That is why `rcdm_image` is built from scratch.
9. **Windows import order.** `datasets`/pyarrow must be imported before torch; otherwise the ImageNet-100 path crashes with exit 139. `src/ijepa/__init__.py` handles this. The exact trigger was not isolated: simple `import torch; import pyarrow` tests passed.
10. **`make_dataset` for STL-10 passes `download=True`.** A missing `data/stl10_binary/` triggers a 2.6 GB download.
11. **Dependencies (updated 2026-10-31).** `requirements.txt` pins the versions on the development machine (Python 3.10), where the 36 CPU tests pass. The Modal images pin their own versions in `modal_app.py` (Python 3.11). `.gitignore` keeps data, checkpoints and run outputs out of Git.
12. **`scripts/visualize_encoder.py` is STL-10-only.** It reads `<data-root>/stl10_binary/class_names.txt`, so ImageNet-100 checkpoints need a change first.
13. **`modal_app.py main()`'s ValueError help text lists only some actions.** The `if/elif` chain is authoritative.
14. **`training.max_hours` measures time since *this process* started.** With Modal retries, each retry gets a fresh budget. The ViT-L H100 run does not use it (`max_hours: 0`); at the user's instruction it relies on the account's Modal budget limit.
15. **Fixed 2026-09-29:** resuming with `epoch_checkpoint_from` used to re-write the resumed epoch's snapshot. Existing snapshots are now skipped (§4.7).
16. **The decoder trainer's resume is not bit-identical.** The oracle run's step-36,001 row read L1 0.1388 after a resume against 0.1371 before it: the same state, different batches.
17. **The completion CLI checks lineage by file name.** Renamed downloads such as `modal_encoder_step_0200000_final.pt` trip it, and `--allow-mismatch` is then needed even for a true pair.
18. **`evaluate_decoder.py` scores the first `--examples` images of the test split in order.** ImageNet-100 validation is sorted by class, so its saved grid shows only class 0 ("bonnet"); the metrics cover all 5,000 images.

## 8. Environment

| Where | Versions |
| --- | --- |
| Local (verified 2026-09-28) | Windows 11, Python 3.10.0, torch 2.5.1+cu121, torchvision 0.20.1+cu121, numpy 2.2.6, PyYAML 6.0.3, datasets 5.0.0, pyarrow 24.0.0, Pillow 10.4.0, modal 1.5.1; NVIDIA RTX 4060 Laptop GPU, 8,188 MiB |
| Local additions (verified 2026-09-29) | diffusers 0.35.2 (for RCDM), graphifyy 0.9.71 (the knowledge-graph tool, not used by the code); modal 1.5.1 unchanged |
| Modal images (`modal_app.py`) | `image`: `debian_slim`, Python 3.11, torch 2.5.1, torchvision 0.20.1, pyyaml 6.0.2, numpy 2.1.3. `hf_image` adds `datasets==5.0.0`; `rcdm_image` adds `datasets==5.0.0` and `diffusers==0.35.2`. Each copies `src/` and `configs/`, and each account's Volume `ijepa-research` is mounted at `/vol`. |
