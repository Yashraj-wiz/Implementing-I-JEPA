import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.ijepa.masks import MultiBlockMaskSampler
from src.ijepa.decoder import MaskedPatchDecoder, compose_masked_images, patchify, unpatchify
from src.ijepa.model import IJEPAModel, gather_tokens, latent_loss
from src.ijepa.train import load_checkpoint, parameter_groups, prepare_resume_state, save_checkpoint
from src.ijepa.evaluate import weighted_knn


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.sampler = MultiBlockMaskSampler(64, 8, min_context_patches=4)
        self.model = IJEPAModel(64, 8, 64, 2, 4, 48, 2, 4, 2.0)

    def test_gather_preserves_order(self):
        tokens = torch.arange(2 * 8 * 3).reshape(2, 8, 3)
        indices = torch.tensor([[3, 1], [7, 0]])
        result = gather_tokens(tokens, indices)
        self.assertTrue(torch.equal(result[0, 0], tokens[0, 3]))
        self.assertTrue(torch.equal(result[1, 1], tokens[1, 0]))

    def test_masks_are_valid_and_context_excludes_targets(self):
        masks = self.sampler.sample(8, torch.Generator().manual_seed(1))
        self.assertEqual(masks.targets.shape[0], 4)
        self.assertTrue(((masks.context >= 0) & (masks.context < 64)).all())
        for batch in range(8):
            target_union = set(masks.targets[:, batch].flatten().tolist())
            self.assertTrue(target_union.isdisjoint(masks.context[batch].tolist()))

    def test_target_has_no_gradient_and_shapes_align(self):
        images = torch.rand(2, 3, 64, 64)
        masks = self.sampler.sample(2, torch.Generator().manual_seed(2))
        prediction, target = self.model(images, masks)
        self.assertEqual(prediction.shape, target.shape)
        loss = latent_loss(prediction, target, "smooth_l1"); loss.backward()
        self.assertTrue(any(p.grad is not None for p in self.model.context_encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.target_encoder.parameters()))

    def test_ema_formula(self):
        before = [p.detach().clone() for p in self.model.target_encoder.parameters()]
        with torch.no_grad():
            for p in self.model.context_encoder.parameters(): p.add_(1.0)
        context = [p.detach().clone() for p in self.model.context_encoder.parameters()]
        self.model.ema_update(0.75)
        for old, source, updated in zip(before, context, self.model.target_encoder.parameters()):
            self.assertTrue(torch.allclose(updated, old * 0.75 + source * 0.25))

    def test_hand_computed_losses(self):
        prediction = torch.tensor([0.0, 2.0]); target = torch.tensor([0.0, 0.0])
        self.assertEqual(float(latent_loss(prediction, target, "l2")), 2.0)
        self.assertEqual(float(latent_loss(prediction, target, "smooth_l1")), 0.75)

    def test_checkpoint_round_trip(self):
        optimizer = torch.optim.AdamW(parameter_groups(self.model, 0.04), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        expected = copy.deepcopy(self.model.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "latest.pt"
            save_checkpoint(checkpoint, self.model, optimizer, scaler, {"global_step": 3}, {"name": "test"})
            with torch.no_grad():
                next(self.model.context_encoder.parameters()).add_(5.0)
            state = load_checkpoint(checkpoint, self.model, optimizer, scaler, torch.device("cpu"))
        self.assertEqual(state["global_step"], 3)
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, expected[name]), name)

    def test_resume_cursor_can_reset_for_new_batch_size(self):
        original = {"global_step": 58_000, "epoch": 37, "batch_in_epoch": 203}
        resumed = prepare_resume_state(original, reset_data_cursor=True)
        self.assertEqual(resumed["global_step"], 58_000)
        self.assertEqual(resumed["epoch"], 37)
        self.assertEqual(resumed["batch_in_epoch"], 0)
        self.assertEqual(original["batch_in_epoch"], 203)

    def test_patchify_round_trip(self):
        images = torch.arange(2 * 3 * 16 * 16, dtype=torch.float32).reshape(2, 3, 16, 16)
        patches = patchify(images, 8)
        self.assertTrue(torch.equal(unpatchify(patches, 8, 16), images))

    def test_completion_preserves_visible_patches_and_averages_overlap(self):
        images = torch.ones(1, 3, 16, 16)
        patch_dim = 3 * 8 * 8
        indices = torch.tensor([[[0, 1]], [[1, 3]]])
        predictions = torch.stack(
            [torch.full((1, 2, patch_dim), 2.0), torch.full((1, 2, patch_dim), 4.0)]
        )
        masked, completed, target_mask = compose_masked_images(images, predictions, indices, 8)
        completed_patches = patchify(completed, 8)
        self.assertTrue(torch.equal(patchify(masked, 8)[0, 2], patchify(images, 8)[0, 2]))
        self.assertTrue(torch.equal(completed_patches[0, 2], patchify(images, 8)[0, 2]))
        self.assertTrue(torch.all(completed_patches[0, 0] == 2.0))
        self.assertTrue(torch.all(completed_patches[0, 1] == 3.0))
        self.assertTrue(torch.all(completed_patches[0, 3] == 4.0))
        self.assertTrue(torch.equal(target_mask, torch.tensor([[True, True, False, True]])))

    def test_completion_accepts_mixed_precision_predictions(self):
        images = torch.ones(1, 3, 16, 16, dtype=torch.float32)
        indices = torch.tensor([[[0, 1]]])
        predictions = torch.ones(1, 1, 2, 3 * 8 * 8, dtype=torch.bfloat16)
        _masked, completed, _target_mask = compose_masked_images(
            images, predictions, indices, 8
        )
        self.assertEqual(completed.dtype, torch.float32)

    def test_decoder_does_not_backpropagate_into_frozen_ijepa(self):
        self.model.requires_grad_(False)
        decoder = MaskedPatchDecoder(64, 8, 64, decoder_dim=48, depth=2, heads=4, mlp_ratio=2.0)
        images = torch.rand(2, 3, 64, 64)
        masks = self.sampler.sample(2, torch.Generator().manual_seed(5))
        with torch.no_grad():
            prediction = self.model.predictions(images, masks)
        reconstructed = decoder(prediction, masks.targets)
        reconstructed.square().mean().backward()
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in decoder.parameters()))

    def test_decoder_conditioning_modes(self):
        from src.ijepa.train_decoder import decoder_conditioning

        self.model.eval()
        images = torch.rand(2, 3, 64, 64)
        masks = self.sampler.sample(2, torch.Generator().manual_seed(6))
        oracle = decoder_conditioning(self.model, images, masks, "oracle", True)
        predicted = decoder_conditioning(self.model, images, masks, "predicted", True)
        self.assertTrue(torch.equal(oracle, self.model.targets(images, masks.targets, True)))
        with torch.no_grad():
            self.assertTrue(torch.equal(predicted, self.model.predictions(images, masks)))
        self.assertEqual(oracle.shape, predicted.shape)
        self.assertFalse(oracle.requires_grad or predicted.requires_grad)
        with self.assertRaises(ValueError):
            decoder_conditioning(self.model, images, masks, "pixels", True)

    def test_mask_collator_matches_sampler_contract(self):
        from src.ijepa.masks import MaskCollator

        collate = MaskCollator(self.sampler)
        batch = [(torch.rand(3, 64, 64), i) for i in range(5)]
        images, labels, masks = collate(batch)
        self.assertEqual(tuple(images.shape), (5, 3, 64, 64))
        self.assertEqual(labels.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(masks.targets.shape[:2], (4, 5))
        for b in range(5):
            self.assertTrue(set(masks.targets[:, b].flatten().tolist()).isdisjoint(masks.context[b].tolist()))

    def test_accumulated_gradient_equals_full_batch_gradient(self):
        from src.ijepa.masks import MaskBatch

        images = torch.rand(4, 3, 64, 64)
        masks = self.sampler.sample(4, torch.Generator().manual_seed(9))
        half = lambda m, sl: MaskBatch(m.context[sl], m.targets[:, sl])

        def grads(loss_fn):
            self.model.zero_grad(set_to_none=True)
            loss_fn()
            return [p.grad.clone() for p in self.model.trainable_parameters()]

        def full():
            latent_loss(*self.model(images, masks), "l2").backward()

        def accumulated(steps=2):
            for sl in (slice(0, 2), slice(2, 4)):
                (latent_loss(*self.model(images[sl], half(masks, sl)), "l2") / steps).backward()

        for a, b in zip(grads(full), grads(accumulated)):
            self.assertTrue(torch.allclose(a, b, atol=1e-6, rtol=1e-4))

    def test_epoch_snapshots_every_n_with_full_checkpoints_are_never_overwritten(self):
        import yaml
        from src.ijepa.train import train

        smoke = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml").read_text())
        step_of = lambda path: torch.load(path, map_location="cpu", weights_only=False)["state"]["global_step"]
        with tempfile.TemporaryDirectory() as directory:
            def run(name, total_steps, **training):
                cfg = copy.deepcopy(smoke)  # 16 synthetic images / batch 4 -> 4 optimizer steps per epoch
                cfg["training"].update({"output_dir": str(Path(directory) / name), "resume": True, "log_every": 1000,
                                        "checkpoint_every": 1000, "total_steps": total_steps, **training})
                path = Path(directory) / f"{name}.yaml"
                path.write_text(yaml.safe_dump(cfg))
                train(str(path))
                return Path(directory) / name

            # Every 2 epochs from epoch 1, with full checkpoints. Stop inside epoch 3 (global steps 13-16).
            every2 = dict(epoch_checkpoint_from=1, epoch_checkpoint_every=2, epoch_checkpoint_full=True)
            out = run("every2", 14, **every2)
            self.assertEqual(sorted(p.name for p in out.glob("epoch_*")),
                             ["epoch_001.pt", "epoch_001_weights.pt", "epoch_003.pt", "epoch_003_weights.pt"])
            self.assertEqual([step_of(out / n) for n in ("epoch_003.pt", "epoch_003_weights.pt")], [13, 13])
            self.assertIn("optimizer", torch.load(out / "epoch_003.pt", map_location="cpu", weights_only=False))
            # Resume inside the already-saved epoch 3 with a changed schedule: every epoch from 4, plus a final
            # snapshot after the last step (24 = end of epoch 5, tagged epoch_006). Epoch 3 must not be overwritten.
            run("every2", 24, **every2, epoch_checkpoint_dense_from=4, epoch_checkpoint_final=True)
            self.assertEqual([step_of(out / n) for n in ("epoch_003.pt", "epoch_003_weights.pt")], [13, 13])
            self.assertEqual({n: step_of(out / f"epoch_{n:03d}.pt") for n in (4, 5, 6)}, {4: 17, 5: 21, 6: 24})
            self.assertEqual({n: step_of(out / f"epoch_{n:03d}_weights.pt") for n in (4, 5, 6)}, {4: 17, 5: 21, 6: 24})
            self.assertFalse(any(out.glob("epoch_002*")))

            # Defaults keep the original behaviour: bf16 weights every epoch, no full checkpoints.
            out = run("default", 10, epoch_checkpoint_from=1)
            self.assertEqual(sorted(p.name for p in out.glob("epoch_*")), ["epoch_001_weights.pt", "epoch_002_weights.pt"])

    def test_decoder_holdout_eval_best_checkpoint_and_early_stop(self):
        import json as _json
        import yaml
        from src.ijepa.train_decoder import split_holdout, train_decoder

        fit, hold = split_holdout(40, 8, seed=3)
        self.assertEqual(len(hold), 8)
        self.assertEqual(sorted(fit + hold), list(range(40)))
        self.assertEqual(split_holdout(40, 8, seed=3), (fit, hold))  # deterministic

        enc_cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml").read_text())
        enc_cfg["data"]["synthetic_count"] = 40
        with tempfile.TemporaryDirectory() as directory:
            enc_path = Path(directory) / "encoder.pt"
            torch.save({"model": IJEPAModel(**enc_cfg["model"]).state_dict(), "config": enc_cfg,
                        "state": {"global_step": 0}}, enc_path)

            def run(name, lr, patience, total_steps=12, masks_in_loader=False):
                cfg = {"seed": 3, "encoder_checkpoint": str(enc_path), "conditioning": "predicted",
                       "data": {"batch_size": 4, "num_workers": 0, "pin_memory": False, "masks_in_loader": masks_in_loader},
                       "decoder": {"decoder_dim": 32, "depth": 1, "heads": 4, "mlp_ratio": 2.0},
                       "optimization": {"lr": lr, "start_lr": lr, "final_lr": lr, "warmup_steps": 0,
                                        "weight_decay": 0.0, "grad_clip": 1.0},
                       "training": {"total_steps": total_steps, "precision": "fp32", "output_dir": str(Path(directory) / name),
                                    "log_every": 1000, "checkpoint_every": 1000, "holdout_size": 8, "eval_every": 2,
                                    "eval_batch_size": 4, "early_stop_patience": patience, "resume": True}}
                path = Path(directory) / f"{name}.yaml"
                path.write_text(yaml.safe_dump(cfg))
                result = train_decoder(str(path))
                rows = [_json.loads(l) for l in (Path(directory) / name / "metrics.jsonl").read_text().splitlines()]
                return result, [r for r in rows if "val_masked_l1" in r]

            # Learning run: an evaluation every 2 steps, best.pt holds the minimum held-out L1.
            result, vals = run("learn", lr=1e-3, patience=0)
            self.assertEqual([r["step"] for r in vals], [2, 4, 6, 8, 10, 12])
            self.assertTrue(Path(result["best_checkpoint"]).exists())
            self.assertAlmostEqual(result["state"]["best_val_masked_l1"], min(r["val_masked_l1"] for r in vals))
            best = torch.load(result["best_checkpoint"], map_location="cpu", weights_only=False)
            self.assertAlmostEqual(best["state"]["best_val_masked_l1"], result["state"]["best_val_masked_l1"])
            # Frozen run (lr 0): the held-out score cannot improve after the first evaluation -> early stop.
            result, vals = run("frozen", lr=0.0, patience=2)
            self.assertTrue(result["early_stopped"])
            self.assertEqual(result["state"]["global_step"], 6)
            self.assertEqual(len({r["val_masked_l1"] for r in vals}), 1)  # fixed masks: identical scores
            # Loader-side mask sampling path runs end to end.
            result, vals = run("collate", lr=1e-3, patience=0, total_steps=4, masks_in_loader=True)
            self.assertEqual(result["state"]["global_step"], 4)

    def test_weighted_knn(self):
        train_features = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
        train_labels = torch.tensor([0, 0, 1, 1])
        test_features = torch.tensor([[1.0, 0.05], [0.05, 1.0]])
        test_labels = torch.tensor([0, 1])
        self.assertEqual(weighted_knn(train_features, train_labels, test_features, test_labels, k=2), 1.0)


class VisualizationPipelineTests(unittest.TestCase):
    """Checks that completion grids show exactly what the model produced, at the real geometry."""

    def setUp(self):
        from src.ijepa.decoder import gather_target_patches

        self.gather_target_patches = gather_target_patches
        self.image_size, self.patch = 224, 16
        grid = self.image_size // self.patch
        self.sampler = MultiBlockMaskSampler(self.image_size, self.patch)  # real mask defaults
        # Every pixel of patch k (row-major) holds the value k, in all three channels.
        index = torch.arange(grid * grid, dtype=torch.float32).reshape(grid, grid)
        image = index.repeat_interleave(self.patch, 0).repeat_interleave(self.patch, 1)
        self.images = torch.stack([image + 1000 * b for b in range(3)]).unsqueeze(1).repeat(1, 3, 1, 1)
        self.masks = self.sampler.sample(3, torch.Generator().manual_seed(11))

    def test_patch_index_maps_to_image_coordinates(self):
        patches = patchify(self.images, self.patch)
        grid = self.image_size // self.patch
        for k in (0, 13, 14, 100, 195):
            row, col = divmod(k, grid)
            block = self.images[0, :, row * 16:(row + 1) * 16, col * 16:(col + 1) * 16]
            self.assertTrue(torch.all(block == k))
            self.assertTrue(torch.all(patches[0, k] == k))

    def test_oracle_pixels_reconstruct_exactly_and_holes_match_targets(self):
        true_patches = self.gather_target_patches(patchify(self.images, self.patch), self.masks.targets)
        masked, completed, target_mask = compose_masked_images(
            self.images, true_patches, self.masks.targets, self.patch
        )
        self.assertTrue(torch.equal(completed, self.images))
        masked_patches = patchify(masked, self.patch)
        for b in range(self.images.size(0)):
            union = set(self.masks.targets[:, b].flatten().tolist())
            self.assertEqual(set(target_mask[b].nonzero().flatten().tolist()), union)
            for k in range(masked_patches.size(1)):
                expected = 0.0 if k in union else float(k + 1000 * b)
                self.assertTrue(torch.all(masked_patches[b, k] == expected), (b, k))

    def test_wrong_predictions_change_only_target_patches(self):
        wrong = self.gather_target_patches(patchify(self.images, self.patch), self.masks.targets) - 7.0
        _masked, completed, target_mask = compose_masked_images(
            self.images, wrong, self.masks.targets, self.patch
        )
        changed = (patchify(completed, self.patch) != patchify(self.images, self.patch)).any(-1)
        self.assertTrue(torch.equal(changed, target_mask))

    def test_denormalize_inverts_training_normalization(self):
        from src.ijepa.data import IMAGENET_MEAN, IMAGENET_STD
        from src.ijepa.train_decoder import denormalize

        pixels = torch.rand(2, 3, 8, 8)
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.assertTrue(torch.allclose(denormalize((pixels - mean) / std), pixels, atol=1e-6))

    def test_sample_grid_layout_is_three_rows_of_images(self):
        from PIL import Image
        from src.ijepa.train_decoder import save_sample_grid

        true_patches = self.gather_target_patches(patchify(self.images, self.patch), self.masks.targets)
        normalized = torch.rand(3, 3, 224, 224)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grid.png"
            save_sample_grid(path, normalized, true_patches, self.masks.targets, self.patch)
            with Image.open(path) as grid_image:
                width, height = grid_image.size
        pad = 2  # torchvision.utils.save_image default padding
        self.assertEqual((width, height), (3 * (224 + pad) + pad, 3 * (224 + pad) + pad))


class RCDMTests(unittest.TestCase):
    """Latent RCDM visualizer (decision D-011): conditioning must actually steer the denoiser."""

    def setUp(self):
        from src.ijepa import rcdm

        self.rcdm = rcdm
        torch.manual_seed(0)
        self.model = rcdm.RCDMUNet(feature_dim=24, latent_size=8, block_out_channels=(32, 64),
                                   layers_per_block=1, attention_levels=(False, True),
                                   attention_head_dim=16, cond_hidden_dim=32)
        self.scheduler = rcdm.make_scheduler()

    def test_output_shape_and_conditioning_changes_prediction(self):
        latents = torch.randn(2, 4, 8, 8)
        t = torch.tensor([10, 500])
        a = self.model(latents, t, torch.randn(2, 24))
        b = self.model(latents, t, torch.randn(2, 24))
        self.assertEqual(a.shape, latents.shape)
        self.assertGreater(float((a - b).abs().mean()), 1e-4)

    def test_loss_is_finite_and_trains_conditioning_path(self):
        loss = self.rcdm.diffusion_loss(self.model, self.scheduler, torch.randn(2, 4, 8, 8), torch.randn(2, 24),
                                        torch.Generator().manual_seed(1))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in self.model.cond_proj.parameters()))

    def test_ddim_sampling_shape(self):
        out = self.rcdm.sample(self.model, torch.randn(3, 24), steps=3,
                               generator=torch.Generator().manual_seed(2), latent_shape=(4, 8, 8))
        self.assertEqual(tuple(out.shape), (3, 4, 8, 8))
        self.assertTrue(torch.isfinite(out).all())

    def test_vae_input_range_conversion(self):
        from src.ijepa.data import IMAGENET_MEAN, IMAGENET_STD

        pixels = torch.rand(2, 3, 4, 4)
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        vae_range = self.rcdm.imagenet_normalized_to_vae_range((pixels - mean) / std)
        self.assertTrue(torch.allclose(vae_range, pixels * 2 - 1, atol=1e-6))


    def test_rcdm_ema_warmup_schedule(self):
        from src.ijepa.train_rcdm import ema_decay_at

        self.assertEqual(ema_decay_at(0, 0.9999, False), 0.9999)
        self.assertAlmostEqual(ema_decay_at(0, 0.9999, True), 0.1)
        self.assertAlmostEqual(ema_decay_at(4990, 0.9999, True), 0.9982, places=4)
        self.assertEqual(ema_decay_at(1_000_000, 0.9999, True), 0.9999)
        values = [ema_decay_at(t, 0.9999, True) for t in range(0, 200_000, 997)]
        self.assertEqual(values, sorted(values))

    def test_rcdm_ema_warmup_tracks_model_early(self):
        """20 CPU steps: with warmup the EMA follows the trained UNet; without it stays at the init."""
        import yaml

        from src.ijepa.train_rcdm import RCDMState, train_rcdm

        model_cfg = {"block_out_channels": [32, 64], "layers_per_block": 1, "attention_levels": [False, True],
                     "attention_head_dim": 16, "cond_hidden_dim": 32}
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache.pt"
            torch.save({"latents": torch.randn(16, 4, 8, 8).half(), "features": torch.randn(16, 24)}, cache)
            ratios = {}
            for warmup in (False, True):
                out = Path(directory) / f"run_{warmup}"
                cfg = {"seed": 3, "cache": {"train": str(cache)}, "model": model_cfg,
                       "optimization": {"lr": 1e-3, "warmup_steps": 0, "ema_decay": 0.9999, "ema_warmup": warmup},
                       "training": {"total_steps": 20, "batch_size": 4, "device": "cpu", "output_dir": str(out),
                                    "log_every": 5, "checkpoint_every": 20, "resume": False}}
                config = Path(directory) / f"rcdm_{warmup}.yaml"
                config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
                train_rcdm(str(config))
                torch.manual_seed(3)
                init = RCDMState({**model_cfg, "feature_dim": 24, "latent_size": 8}).model.state_dict()
                weights = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)["model"]
                keys = [k for k in init if init[k].is_floating_point()]
                moved = sum(float((weights[f"ema.{k}"] - init[k]).norm()) for k in keys)
                trained = sum(float((weights[f"model.{k}"] - init[k]).norm()) for k in keys)
                ratios[warmup] = moved / trained
                rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
                self.assertIn("ema_decay", rows[-1])
            self.assertLess(ratios[False], 0.01)
            self.assertGreater(ratios[True], 0.5)

class RCDMEvaluationTests(unittest.TestCase):
    """scripts/eval_rcdm.py metric helpers: k-NN predictions and source-retrieval ranks."""

    def test_knn_predictions_match_weighted_knn_and_ranks_are_exact(self):
        from scripts.eval_rcdm import knn_predict, source_ranks
        from src.ijepa.evaluate import weighted_knn

        generator = torch.Generator().manual_seed(0)
        bank, bank_labels = torch.randn(300, 16, generator=generator), torch.randint(0, 5, (300,), generator=generator)
        queries, labels = torch.randn(70, 16, generator=generator), torch.randint(0, 5, (70,), generator=generator)
        predictions = knn_predict(bank, bank_labels, queries, batch_size=32)
        self.assertEqual(int((predictions == labels).sum()), round(weighted_knn(bank, bank_labels, queries, labels) * 70))

        gallery = torch.eye(4)
        queries = torch.tensor([[1.0, 0, 0, 0], [0.9, 0.5, 0, 0], [0.9, 0.5, 0, 0]])
        ranks = source_ranks(queries, gallery, torch.tensor([0, 1, 0]))
        self.assertEqual(ranks.tolist(), [1, 2, 1])

class CompletionCliTests(unittest.TestCase):
    """src/ijepa/complete.py: preprocessing, rectangle mapping, checkpoint checks, determinism, outputs."""

    def setUp(self):
        import yaml
        from PIL import Image

        from src.ijepa import complete

        self.complete = complete
        self.enc_cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml").read_text())
        self.image = Image.fromarray((torch.rand(240, 320, 3) * 255).to(torch.uint8).numpy())  # W=320, H=240

    def _checkpoints(self, directory, encoder_name="encoder.pt", conditioning="predicted", encoder_dim=None):
        enc = Path(directory) / encoder_name
        torch.save({"model": IJEPAModel(**self.enc_cfg["model"]).state_dict(), "config": self.enc_cfg,
                    "state": {"global_step": 7}}, enc)
        dec_cfg = {"decoder_dim": 32, "depth": 1, "heads": 4, "mlp_ratio": 2.0}
        decoder = MaskedPatchDecoder(64, 8, encoder_dim or self.enc_cfg["model"]["encoder_dim"], **dec_cfg)
        dec = Path(directory) / "decoder.pt"
        torch.save({"model": decoder.state_dict(), "state": {"global_step": 3},
                    "config": {"encoder_checkpoint": f"/vol/somewhere/{encoder_name}", "conditioning": conditioning,
                               "decoder": dec_cfg}}, dec)
        return enc, dec

    def test_preprocess_matches_training_eval_transform_and_rect_mapping(self):
        from src.ijepa.data import _eval_transform

        tensor, geo = self.complete.preprocess(self.image, 64)
        self.assertTrue(torch.equal(tensor[0], _eval_transform({"image_size": 64})(self.image)))
        # 320x240 -> Resize(96) -> 128x96 -> CenterCrop(64) at left 32, top 16 (scale 0.4).
        self.assertEqual((geo["resized_size"], geo["crop_left"], geo["crop_top"]), ([128, 96], 32, 16))
        to_idx = self.complete.rect_to_patch_indices
        self.assertEqual(to_idx((80, 40, 100, 60), geo, 8), [0])               # original px -> crop [0,8)x[0,8)
        self.assertEqual(to_idx((0, 0, 8, 8), geo, 8, space="crop"), [0])
        self.assertEqual(to_idx((8, 0, 17, 8), geo, 8, space="crop"), [1, 2])  # touching a patch masks it
        self.assertEqual(len(to_idx((0, 0, 320, 240), geo, 8)), 64)            # whole image -> every patch
        with self.assertRaises(self.complete.CompletionError):
            to_idx((0, 0, 20, 20), geo, 8)                                      # outside the centre crop
        with self.assertRaises(self.complete.CompletionError):
            self.complete.rect_mask_batch(list(range(64)), 64)                  # nothing left visible
        masks = self.complete.rect_mask_batch([0, 1, 9], 64, context_tokens=10, seed=1)
        self.assertEqual(masks.targets.tolist(), [[[0, 1, 9]]])
        self.assertEqual(masks.context.size(1), 10)
        self.assertTrue(set(masks.context[0].tolist()).isdisjoint({0, 1, 9}))

    def test_random_masks_are_seeded(self):
        sample = lambda seed: self.complete.random_mask_batch(self.enc_cfg["mask"], 64, 8, seed)
        a, b, c = sample(5), sample(5), sample(6)
        self.assertTrue(torch.equal(a.targets, b.targets) and torch.equal(a.context, b.context))
        self.assertFalse(torch.equal(a.targets, c.targets) and torch.equal(a.context, c.context))

    def test_checkpoint_compatibility_checks(self):
        load = self.complete.load_models
        cpu = torch.device("cpu")
        with tempfile.TemporaryDirectory() as directory:
            enc, dec = self._checkpoints(directory)
            load(str(enc), str(dec), cpu)                                   # same encoder file name: accepted
            other = Path(directory) / "other_encoder.pt"
            other.write_bytes(enc.read_bytes())
            with self.assertRaises(self.complete.CompletionError):
                load(str(other), str(dec), cpu)                             # trained with a different encoder
            load(str(other), str(dec), cpu, allow_mismatch=True)
        with tempfile.TemporaryDirectory() as directory:
            enc, dec = self._checkpoints(directory, conditioning="oracle")
            with self.assertRaises(self.complete.CompletionError):
                load(str(enc), str(dec), cpu)                               # oracle decoders read the hidden region
        with tempfile.TemporaryDirectory() as directory:
            enc, dec = self._checkpoints(directory, encoder_dim=48)
            with self.assertRaises(self.complete.CompletionError):
                load(str(enc), str(dec), cpu)                               # wrong token width

    def test_cli_writes_outputs_and_preserves_visible_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            enc, dec = self._checkpoints(directory)
            image_path = Path(directory) / "photo.png"
            self.image.save(image_path)
            for extra, name in (([], "random"), (["--rect", "100", "60", "180", "140"], "rect")):
                out = Path(directory) / name
                args = self.complete.build_parser().parse_args(
                    ["--image", str(image_path), "--encoder", str(enc), "--decoder", str(dec), "--output-dir", str(out),
                     "--device", "cpu", "--seed", "3", *extra])
                meta = self.complete.run(args)
                for file in ("original.png", "masked.png", "completed.png", "comparison.png", "metadata.json"):
                    self.assertTrue((out / file).exists(), file)
                self.assertTrue(meta["visible_pixels_exact"])
                self.assertEqual(meta["masked_patches"] + meta["context_patches"] <= 64, True)
                self.assertEqual(self.complete.run(args)["masked_l1"], meta["masked_l1"])  # deterministic
            self.assertEqual(json.loads((Path(directory) / "rect" / "metadata.json").read_text())["mode"], "rect")



class _FakeVAE(torch.nn.Module):
    """Deterministic stand-in for the SD-VAE (8x downsampling, 4 latent channels) so tests need no download."""

    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"scaling_factor": 0.18215})()
        self.dtype = torch.float32
        torch.manual_seed(5)
        self.down, self.up = torch.nn.Conv2d(3, 4, 8, stride=8), torch.nn.ConvTranspose2d(4, 3, 8, stride=8)

    def encode(self, x):
        return type("Out", (), {"latent_dist": type("Dist", (), {"mean": self.down(x)})()})()

    def decode(self, z):
        return type("Out", (), {"sample": torch.tanh(self.up(z))})()


class RCDMCompletionTests(unittest.TestCase):
    """complete.py --rcdm: no information from the hidden region may reach the completion."""

    def setUp(self):
        import yaml

        from src.ijepa import complete
        from src.ijepa.rcdm import RCDMUNet

        self.complete = complete
        cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml").read_text())
        torch.manual_seed(0)
        self.encoder = IJEPAModel(**cfg["model"]).eval()
        self.mask_cfg = cfg["mask"]
        self.unet_cfg = {"feature_dim": 64, "latent_size": 8, "block_out_channels": [32, 64], "layers_per_block": 1,
                         "attention_levels": [False, True], "attention_head_dim": 16, "cond_hidden_dim": 32}
        self.rcdm = RCDMUNet(**self.unet_cfg).eval()
        self.vae = _FakeVAE().eval()
        self.image = torch.randn(1, 3, 64, 64)

    def _complete(self, image, masks, conditioning):
        return self.complete.complete_rcdm(self.encoder, self.rcdm, self.vae, image, masks, 8, torch.device("cpu"),
                                           conditioning, samples=2, steps=3, seed=4)

    def test_hidden_pixels_never_reach_the_completion(self):
        for masks in (self.complete.random_mask_batch(self.mask_cfg, 64, 8, seed=2),
                      self.complete.rect_mask_batch([9, 10, 17, 18], 64)):
            hidden = torch.zeros(64)
            hidden[self.complete.hidden_patches(masks)] = 1
            pixel = hidden.view(1, 1, 8, 8).repeat_interleave(8, 2).repeat_interleave(8, 3).bool()
            other = torch.where(pixel, torch.randn_like(self.image) * 3, self.image)   # differs only in the hole
            for conditioning in ("predicted", "visible"):
                a, b = self._complete(self.image, masks, conditioning), self._complete(other, masks, conditioning)
                self.assertTrue(torch.equal(a["completed"], b["completed"]), conditioning)
                self.assertTrue(a["visible_pixels_exact"])
                self.assertEqual(len(a["masked_l1_per_sample"]), 2)
            oracle_a = self._complete(self.image, masks, "oracle")["completed"]
            oracle_b = self._complete(other, masks, "oracle")["completed"]
            self.assertFalse(torch.equal(oracle_a, oracle_b))                       # the oracle does read the hole

    def test_inpaint_keeps_known_cells_and_condition_shapes(self):
        from src.ijepa.rcdm import inpaint

        known = torch.randn(2, 4, 8, 8)
        keep = torch.zeros(1, 1, 8, 8)
        keep[..., :4, :] = 1
        out = inpaint(self.rcdm, torch.randn(2, 64), known, keep, steps=3, generator=torch.Generator().manual_seed(0))
        self.assertTrue(torch.equal(out[..., :4, :], known[..., :4, :]))
        self.assertFalse(torch.equal(out[..., 4:, :], known[..., 4:, :]))
        masks = self.complete.random_mask_batch(self.mask_cfg, 64, 8, seed=1)
        for mode in ("predicted", "visible", "oracle"):
            self.assertEqual(tuple(self.complete.rcdm_condition(self.encoder, self.image, masks, 64, mode).shape), (1, 64))

    def test_load_rcdm_refusals(self):
        from src.ijepa.train_rcdm import RCDMState

        cpu = torch.device("cpu")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rcdm.pt"
            torch.save({"model": RCDMState(self.unet_cfg).state_dict(), "state": {"global_step": 1},
                        "config": {"model": self.unet_cfg, "diffusion": {"num_train_timesteps": 1000}}}, path)
            self.complete.load_rcdm(str(path), 64, cpu)
            with self.assertRaises(self.complete.CompletionError):
                self.complete.load_rcdm(str(path), 1024, cpu)                       # feature width mismatch
            with self.assertRaises(self.complete.CompletionError):
                self.complete.load_rcdm(str(path), 64, cpu, "oracle")              # oracle needs --allow-oracle
            self.complete.load_rcdm(str(path), 64, cpu, "oracle", allow_oracle=True)
            bad = Path(directory) / "decoder.pt"
            torch.save({"model": {}, "config": {"decoder": {}}}, bad)
            with self.assertRaises(self.complete.CompletionError):
                self.complete.load_rcdm(str(bad), 64, cpu)

if __name__ == "__main__": unittest.main()
