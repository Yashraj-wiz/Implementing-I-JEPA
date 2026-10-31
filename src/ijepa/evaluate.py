from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import make_dataset
from .model import IJEPAModel


@torch.inference_mode()
def extract_features(encoder, loader, device: torch.device):
    encoder.eval()
    features, labels = [], []
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        representation = encoder(images).mean(dim=1)
        features.append(F.normalize(representation.float(), dim=-1).cpu())
        labels.append(target.long())
    return torch.cat(features), torch.cat(labels)


@torch.inference_mode()
def weighted_knn(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    k: int = 20,
    temperature: float = 0.07,
    batch_size: int = 256,
) -> float:
    classes = int(max(train_labels.max(), test_labels.max()).item()) + 1
    correct = 0
    train_features = F.normalize(train_features.float(), dim=-1)
    test_features = F.normalize(test_features.float(), dim=-1)
    for start in range(0, len(test_features), batch_size):
        query = test_features[start : start + batch_size]
        similarity = query @ train_features.T
        values, indices = similarity.topk(min(k, len(train_features)), dim=1)
        neighbor_labels = train_labels[indices]
        weights = (values / temperature).exp()
        votes = torch.zeros(query.size(0), classes)
        votes.scatter_add_(1, neighbor_labels, weights)
        prediction = votes.argmax(dim=1)
        correct += int((prediction == test_labels[start : start + batch_size]).sum())
    return correct / len(test_labels)


def evaluate_stl10(
    checkpoint_path: str,
    data_root: str = "/vol/data",
    batch_size: int = 256,
    output_path: str | None = None,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    model = IJEPAModel(**cfg["model"])
    model.load_state_dict(checkpoint["model"])
    encoder = model.target_encoder.to(device)

    data_cfg = {**cfg["data"], "root": data_root, "num_workers": 8}
    train_set = make_dataset(data_cfg, train=False, split="train")
    test_set = make_dataset(data_cfg, train=False, split="test")
    loader_args = dict(batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    train_features, train_labels = extract_features(encoder, DataLoader(train_set, **loader_args), device)
    test_features, test_labels = extract_features(encoder, DataLoader(test_set, **loader_args), device)
    accuracy = weighted_knn(train_features, train_labels, test_features, test_labels)
    result = {
        "checkpoint": checkpoint_path,
        "step": checkpoint["state"]["global_step"],
        "knn_k": 20,
        "stl10_test_accuracy": accuracy,
        "train_examples": len(train_labels),
        "test_examples": len(test_labels),
    }
    output = (
        Path(output_path)
        if output_path is not None
        else Path(checkpoint_path).parent / "knn_eval.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
    return result
