# Newcomer FAQ: I-JEPA, from the paper to this code

This guide is for a student who already knows Transformers and the Vision Transformer (ViT) and is new to this project. It works through the questions you are likely to ask, in order:
- what the paper proposes;
- how this repo implements it;
- what was run and what the results mean;
- where to look next.

Facts here come from the paper extract (`paper.md`), the code in `src/ijepa/`, and the project records (`work.md`, `docs/EXPERIMENTS.md`, `docs/ARCHITECTURE.md`). Numbers are measured unless marked *estimate*.

> The public repository includes the code, `README.md`, `docs/ARCHITECTURE.md` and this FAQ. `paper.md` (a text extract of arXiv:2301.08243v3), `work.md`, `task.md`, `CLAUDE.md`, `ijepa.md`, `docs/EXPERIMENTS.md` and `docs/IMAGENET100_RUNBOOK.md` are the project's working records and are kept private; references to them show where each fact came from.

---

## Part 1 — The idea (the paper)

### Q1. What problem is I-JEPA trying to solve?

**Self-supervised learning.** The goal is to learn good image representations from unlabeled images, so that a frozen encoder can later be used for classification, retrieval and similar tasks with few or no labels.

The paper (Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture", arXiv:2301.08243) positions I-JEPA between two existing families:

| Family | Examples | What it learns from | Weakness the paper points to |
| --- | --- | --- | --- |
| **Invariance / joint-embedding** | DINO, iBOT, SimCLR-style | Two augmented views of one image should have similar embeddings | Needs hand-crafted augmentations (crops, colour jitter, …) that inject human bias |
| **Generative / reconstruction** | MAE, BEiT | Reconstruct masked **pixels** (or pixel tokens) | Spends capacity on low-level pixel detail; representations tend to need fine-tuning |
| **I-JEPA** | this paper | Predict the **representations** of masked regions from the visible context | Neither pixel reconstruction nor augmentation invariance |

### Q2. What does "Joint-Embedding Predictive Architecture" mean in plain words?

Take one image and hide some rectangular regions (the **targets**). Show the model the rest (the **context**), and ask it to predict *what an encoder would output* for the hidden regions: their embeddings, not their pixels.

- Pixel detail such as exact texture and noise is hard to predict and often irrelevant.
- Predicting in embedding space lets the model skip that detail and focus on semantic content ("there is a dog's head here").

### Q3. What are the three networks?

All three are Transformers operating on 16×16 patches, as in ViT.

| Network | Input | Output | Trained how |
| --- | --- | --- | --- |
| **Context encoder** f_θ | Only the visible (context) patch tokens | One embedding per context patch | By gradient descent |
| **Target encoder** f_θ̄ | The **full image** (all patches) | One embedding per patch; only the target positions are used | **Not** by gradients: its weights are an **exponential moving average (EMA)** of the context encoder |
| **Predictor** g_φ | Context embeddings + learned "mask tokens" carrying the *positions* of a target block | Predicted embeddings for that target block | By gradient descent |

The loss compares the predictor's output with the target encoder's embeddings at the same positions. In the paper this is an L2 (squared-error) loss averaged over the target blocks (`paper.md:275`).

### Q4. Why is there a second (EMA) encoder? Why not use the context encoder for targets too?

If the same network produced both prediction and target and gradients flowed through both, the easiest solution would be **representation collapse**: output a constant vector for every patch, so the loss drops to zero while learning nothing.

The EMA target encoder is one of the standard ways to avoid this:
- It receives no gradients (`requires_grad_(False)` in the code).
- It changes slowly: θ̄ ← m·θ̄ + (1−m)·θ, with momentum m going from 0.996 to 1.0 during training (`paper.md:958-959`).

Q14 shows how collapse is monitored in this project.

### Q5. How are the masks chosen, and why does it matter?

The mask design is a key part of the paper (`paper.md:960-965`). Per image:
- **4 target blocks**, each covering 15–20% of the image, aspect ratio 0.75–1.5. The blocks are large, so predicting them needs semantic understanding, not copying neighbouring pixels.
- **1 context block** covering 85–100% of the image, square. The target regions are then **removed** from it, so the context never contains the answer.

The context encoder sees only what is left. The target encoder sees the whole image, and its outputs at the 4 target blocks are the prediction targets.

### Q6. Why predict *after* the target encoder instead of masking its input?

The target encoder processes the **full** image, and the targets are taken from its *output* ("masking at the output of the target encoder"). Its target embeddings therefore see the whole image context, which makes them richer targets than encodings of isolated crops.

### Q7. How does the paper evaluate the representations?

Mainly with **frozen-encoder** protocols: linear probing on ImageNet, low-shot (1%) classification, transfer tasks, and some local tasks such as object counting on CLEVR (`paper.md:985-995`).

The paper does not train a pixel decoder for its main results. To *visualize* what the representations contain, it trains a separate diffusion model (RCDM) conditioned on the average-pooled representation (`paper.md:1151-1168`).

---

## Part 2 — This implementation

### Q8. What exactly is in this repo?

- **A from-scratch PyTorch implementation of I-JEPA pretraining** in `src/ijepa/`:
  - `model.py`: ViT, predictor, `IJEPAModel`;
  - `masks.py`: multi-block mask sampler;
  - `train.py`: training loop, schedules, checkpoints;
  - `data.py`: STL-10 and ImageNet-100 loaders;
  - `evaluate.py`: k-NN evaluation.
- **Two ways to turn representations back into pictures:**
  - **`decoder.py` / `train_decoder.py`:** a deterministic *masked-patch decoder*, a project addition that maps predicted target embeddings to RGB patches for masked-image completion;
  - **`rcdm.py` / `train_rcdm.py`:** an *RCDM-style* visualizer, a diffusion model in Stable-Diffusion-VAE latent space conditioned on the pooled features. It mirrors the paper's visualization method. It was trained on ViT-L features (ImageNet-100, 100k steps) on 2026-10-02. Its samples keep the object type, colour, texture and pose of the source image but not faces or fine detail. Measured with `scripts/eval_rcdm.py`: each sample is recognised as its own source image among 5,000 (100% top-1), and samples keep 85% of the real images' k-NN class accuracy (51.8% vs 61.0%).
- **Cloud and ops tooling:**
  - `modal_app.py`: Modal cloud-GPU functions, one per `--action`;
  - `configs/`: one versioned YAML per run;
  - `tests/test_core.py`: 36 unit tests.

The internship brief is `ijepa.md`.

### Q9. Walk me through one training step in the code.

`src/ijepa/train.py` → `IJEPAModel.forward` in `src/ijepa/model.py`:

1. **Sample masks.** `MultiBlockMaskSampler.sample()` returns `MaskBatch(context [B, Nc], targets [4, B, Nt])`. Each entry is a patch index on the 14×14 grid (224 px / 16 px).
2. **Targets (no gradient).** The target encoder runs on the full image and outputs `[B, 196, D]`. Then:
   - apply LayerNorm over the feature dimension;
   - gather the tokens at each target block, giving `[4, B, Nt, D]`.
3. **Context.** Patchify, add positional embeddings, then keep only the context indices and run the context encoder, giving `[B, Nc, D]`.
4. **Predict.** For each target block, the predictor:
   - projects the context tokens to its smaller width;
   - appends `Nt` learned mask tokens plus the target positions;
   - runs its Transformer blocks;
   - takes the last `Nt` outputs and projects them back to width D.
5. **Loss.** `l2` (the paper's) or `smooth_l1` (the released code's choice), in `latent_loss()`.
6. **Update.** AdamW updates the context encoder and predictor only, with gradient clipping. **Then** the EMA update moves the target encoder toward the context encoder.

Schedules: LR warmup then cosine decay; weight decay 0.04 → 0.4; EMA momentum 0.996 → 1.0.

Tensor shapes and every detail are in `docs/ARCHITECTURE.md` §3–§4.

### Q10. If you know ViT, what is different here?

| ViT classifier | I-JEPA encoder here |
| --- | --- |
| `[cls]` token + classification head | **No `[cls]` token.** The output is one embedding per patch, and evaluation averages them. |
| Learned positional embeddings | Fixed 2-D sin-cos positional embeddings |
| Every patch goes in | The context encoder gets a **subset** of patches: tokens are gathered *after* adding positions, so each keeps its true location |
| Trained with labels | Trained with no labels at all |

The **predictor** is itself a smaller ViT that works on sequences mixing real context tokens and "mask tokens" that only carry a position.

### Q10a. Which ViT is used, and how big are its embeddings?

The main model is a **ViT-L/16**: embedding dimension **1024**, 24 blocks, 16 heads, MLP ratio 4, and 303.1M parameters per encoder (the target encoder is an EMA copy). Its predictor is 384 wide and 12 blocks deep, and projects back to 1024.
- A 224 px image gives 14 × 14 = **196 tokens**, so one image is 196 × 1024 numbers at the encoder output.
- The pooled embedding (k-NN, linear probe, RCDM) is a single 1024-dim vector.

Other standard sizes, for comparison:

| Model | Embedding dim | Blocks | Heads | MLP hidden | Encoder parameters | Source |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ViT-Tiny/16 | 192 | 12 | 3 | 768 | ≈5.7M | standard (DeiT); not in this repo |
| ViT-Small/16 | 384 | 12 | 6 | 1536 | 21,589,632 | this repo's STL-10 encoder [C] |
| ViT-Base/16 | 768 | 12 | 12 | 3072 | ≈85.6M | this repo's ViT-B config, computed (`work.md`, 2026-09-28) |
| **ViT-Large/16** (main model) | **1024** | 24 | 16 | 4096 | **303,098,880** | this repo's checkpoint [C2] |

At 224 px with 16 px patches every size gives 196 tokens: per-patch embeddings are 196 × dim and the pooled embedding is one dim-sized vector.

A decoder trained for one width cannot read another, so a ViT-Ti/S/B encoder needs its own decoder (192/384/768-dim input).

### Q11. Where does this implementation deviate from the paper, and why?

The full table is in `docs/ARCHITECTURE.md` §6. The main points:
- **Scale.** The paper uses ImageNet-1K, ViT-B up to ViT-H/G, batch 2048 and 300+ epochs. This project uses STL-10 with ViT-S/16, then **ImageNet-100 with ViT-L/16** at effective batch 512, within a student compute budget.
- **The LR is scaled linearly.** The paper's 1e-3 at batch 2048 becomes 2.5e-4 at batch 512.
- **Loss.** The STL-10 runs used smooth L1, following the released code. The ViT-L run uses the paper's L2.
- **One mask shape per batch.** Each context is trimmed to the batch-minimum length so contexts can be stacked. That is how the paper's "same context size on a GPU" constraint is implemented.
- **The masked-patch decoder is a project addition**, not part of the paper.

### Q12. Why does the context get shorter when the batch gets bigger?

All contexts in a batch must have the same length to form one tensor, so each is randomly subsampled to the **shortest** one in the batch. Measured mean context tokens:

| Batch | Mean context tokens |
| ---: | ---: |
| 1 | ≈90 |
| 64 | ≈60 |

This matters for any single-image tool: it would feed the predictor longer contexts than it saw in training. See `docs/ARCHITECTURE.md` §4.1.

### Q13. What is the masked-patch decoder, and why are its images blurry?

It maps the **predicted** target embeddings to 16×16×3 pixel patches, one target block at a time. The loss is L1 in ImageNet-normalized pixel space. Visible patches are copied unchanged, so only the holes are "invented".

**Why blurry:**
- Deterministic regression averages over plausible fills.
- The decoder sees no visible pixels.
- The main cause was found with an **oracle test**: the same decoder was trained on the *target encoder's* true tokens instead of the predictor's guesses.
  - On a fixed benchmark of 1,024 test images, masked L1 is **0.147** for oracle tokens against **0.531** for predicted tokens, and the oracle fills are near-faithful.
  - So the blur comes mostly from the **predictor's features**, not from the decoder design.
  - Oracle tokens see the hidden region, so this is an upper bound.
- **For the ViT-L encoder:** decoder v1 scores a masked L1 of 0.5415 on all 5,000 ImageNet-100 validation images. Colours, textures, and coarse shapes continue well; faces and legs blur (`artifacts/cli_demo_vitl/gallery.png`).

### Q13a. Which decoders exist, and which images did each one see?

- **Decoder v1** is the ViT-L masked-patch decoder (`artifacts/vitl_decoder/best.pt`). It reads the predictor's tokens and gives one blurry fill.
- **The RCDM** (`artifacts/rcdm_vitl16/step_0100000.pt`) is the diffusion visualizer. It reads one pooled feature and gives several sharp samples.
- The older **STL-10 decoders** (300k, adapted, oracle diagnostic) belong to the ViT-S encoder.

| Model | Trained on | Training-split images it never saw | Code |
| --- | --- | --- | --- |
| I-JEPA ViT-L encoder | all 126,689 ImageNet-100 training images (random crops) | none | `train.py:99` |
| Decoder v1 (masked-patch, `artifacts/vitl_decoder/best.pt`) | 121,689 training images | **5,000** (a fixed random slice, `holdout_size: 5000`, `holdout_seed: 0`): seen by the encoder, not by the decoder; used only to choose `best.pt` | `train_decoder.py:184-193` |
| RCDM (`artifacts/rcdm_vitl16/step_0100000.pt`) | random draws from all 126,689 cached training images, ≈50 per image over 100k steps | none in practice (P(an image never drawn) ≈ e^-50) | `train_rcdm.py:108` |

The 5,000 ImageNet-100 **validation** images were seen by none of the three. Every benchmark and gallery so far (decoder v1 validation L1 0.5415, the RCDM evaluation, both CLI galleries) uses validation or STL-10 test images.

**Why it matters:** a score on images a model trained on can hide memorisation. Decoder v1 scores 0.513 on training batches, 0.5249 on its held-out slice and 0.5415 on validation; gaps that small mean it is not memorising.

### Q14. How do I know training isn't collapsing?

`train.py` logs two statistics of the normalized target features at every log step:

| Metric | Healthy | Sign of collapse |
| --- | --- | --- |
| `feature_std`: std across the batch | Clearly above 0 | Falls toward 0 |
| `mean_cosine`: average similarity between different images | Low | Rises toward 1 |

In the ViT-L run, mean cosine fell to ≈0.11 early, drifted up to ≈0.47 by epoch 67 (a *partial* collapse trend), then fell back to ≈0.22 by epoch 125 as the LR decayed, and stayed there to the end of training at epoch 150.

The loss is **not** a quality signal: it rose while representations improved. Judge quality by evaluation (Q15).

### Q15. How are the representations evaluated here?

**Weighted k-nearest-neighbours (k = 20, temperature 0.07)** on frozen, average-pooled, L2-normalized target-encoder features (`src/ijepa/evaluate.py`). Each test image votes by its 20 most similar training images. No training is involved, which makes it a quick check. Measured results:

| Encoder | Data | Accuracy |
| --- | --- | ---: |
| ViT-S/16, step 200k | STL-10 test | 80.8% |
| ViT-L/16, epoch 60 | ImageNet-100 val | 48.82% |
| ViT-L/16, epoch 68 | ImageNet-100 val | 51.12% |
| ViT-L/16, epoch 75 | ImageNet-100 val | 52.94% |
| **ViT-L/16, epoch 150 (final)** | ImageNet-100 val | **60.02%** |
| ViT-L/16, random init | ImageNet-100 val | 8.66% |

Chance on ImageNet-100 is 1%. A **linear probe** (the paper's main protocol) on the same frozen epoch-150 features gives **65.24% top-1** (88.00% top-5) after an extended learning-rate sweep, against 16.96% for a random-init ViT-L under the same sweep. An MAE baseline is **not done yet** (`task.md` §8).

### Q16. Why ImageNet-100 at 160 px, and not full ImageNet?

- **Budget.** The project runs on a laptop GPU (8 GB) plus limited Modal credit.
- **What the copy is.** ImageNet-100 is the 100-class CMC subset (126,689 train / 5,000 validation images, Hugging Face `clane9/imagenet-100`).
- **The 160 px catch.** This copy is pre-resized to a 160 px short side, so every 224 px crop is upsampled. Results are therefore not directly comparable with full-resolution numbers in papers.
- **Licence.** Non-commercial research and educational use only.

---

## Part 3 — Working on the project

### Q17. Which documents should I read, and in what order?

1. `README.md`: what this is and the main commands.
2. **This file.**
3. `docs/ARCHITECTURE.md`: exact tensor contracts, deviations, and code caveats.
4. `docs/EXPERIMENTS.md`: every run, checkpoint, and result.
5. `task.md`: what is left to do.
6. `work.md`: the full chronological log, including decisions D-001…D-012 and why they were made. Append to it; never rewrite it.
7. `docs/IMAGENET100_RUNBOOK.md`: how the ViT-L run on Modal is operated.
8. `CLAUDE.md`: condensed rules and gotchas, also used by Claude Code.

### Q18. How do I run something small to check my setup?

From the repo root:

```powershell
python -m unittest discover -s tests -v              # 36 CPU tests, under a minute
python -m src.ijepa.train --config configs/smoke.yaml  # 8-step tiny synthetic run
```

On Windows:
- New scripts must `import src.ijepa` **before** torch, because `datasets` must load first or the process can segfault.
- To hide the GPU, use `CUDA_VISIBLE_DEVICES=-1`.

### Q19. How do configs and checkpoints work?

- **Configs.** Every run is driven by one YAML with the sections `seed, data, model, mask, objective, optimization, training`.
- **Resuming.** With `resume: true`, rerunning the same config **continues** from `<output_dir>/latest.pt`. So a new experiment needs a new `output_dir`.
- **Checkpoint contents.** A checkpoint holds the model (both encoders + predictor), the optimizer, the step counters, the RNG state and the config, so models can always be rebuilt from the file itself.
- **Weights-only snapshots.** Files like `epoch_NNN_weights.pt` are bf16 weights for evaluation; they cannot resume training.

### Q20a. How do I complete my own image?

```powershell
python -m src.ijepa.complete --image photo.jpg --encoder artifacts/vitl_snapshots/epoch_150_weights.pt --decoder artifacts/vitl_decoder/best.pt --output-dir out/ [--rect X0 Y0 X1 Y1]
```

- **Masks:** without `--rect` it uses random training-style masks; with it, it masks the rectangle, given in your image's pixels.
- **Outputs:** `original/masked/completed/comparison.png` and `metadata.json`. Visible pixels are kept exactly.
- **Safety check:** the CLI refuses a decoder that was trained with a different encoder.
- **Generative version:** replace `--decoder ...` with `--rcdm artifacts/rcdm_vitl16/step_0100000.pt` (optionally `--vae-cache <Hugging Face cache dir>`) to get several sharp samples (`completed_1..4.png`) instead of one blurry average. It builds RCDM's input from the visible pixels only (visible tokens + predictor tokens), so it never peeks at the hole. Samples score worse on pixel L1 than the decoder (0.160 vs 0.108 over 20 gallery images, `artifacts/cli_demo_rcdm/` and `artifacts/cli_demo_set2/`) because L1 rewards the average.

### Q20. What are sensible first questions to explore yourself?

- Why does the target encoder see the full image while the context encoder does not? Try `tests/test_core.py::test_masks_are_valid_and_context_excludes_targets`.
- What happens to `mean_cosine` if the EMA momentum is set to 0, so the target equals the context encoder? Expect collapse pressure; try it only on the smoke config.
- How much does the predictor's width or depth matter? The paper uses a narrow predictor; compare ViT-S (192 wide) with ViT-L (384 wide) in the configs.
- Why is k-NN on the **target** encoder, while the decoder uses the **predictor's** outputs? (Q13, Q15)
- How would a linear probe differ from k-NN, and why is it the paper's main metric? (`task.md` §8)
