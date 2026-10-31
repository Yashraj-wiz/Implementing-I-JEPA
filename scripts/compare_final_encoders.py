from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `src` imports work

from src.ijepa.evaluate import evaluate_stl10


def main() -> None:
    artifacts = Path("artifacts")
    checkpoints = {
        "laptop_step_200000": artifacts
        / "stl10_vits16_local"
        / "step_0200000.pt",
        "modal_step_200000": artifacts / "modal_encoder_step_0200000_final.pt",
    }
    results = {}
    for name, checkpoint in checkpoints.items():
        results[name] = evaluate_stl10(
            str(checkpoint),
            data_root="data",
            batch_size=128,
            output_path=str(artifacts / f"knn_eval_{name}.json"),
        )
    comparison = {
        "protocol": {
            "dataset": "STL-10",
            "reference_examples": 5000,
            "test_examples": 8000,
            "knn_k": 20,
            "feature_batch_size": 128,
        },
        "results": results,
    }
    output = artifacts / "encoder_step200k_comparison.json"
    output.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison), flush=True)


if __name__ == "__main__":
    main()
