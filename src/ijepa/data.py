from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class FixedSyntheticImages(Dataset):
    def __init__(self, count: int, image_size: int, seed: int = 0) -> None:
        generator = torch.Generator().manual_seed(seed)
        low_res = torch.rand(count, 3, 8, 8, generator=generator)
        self.images = torch.nn.functional.interpolate(
            low_res, size=(image_size, image_size), mode="bilinear", align_corners=False
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], -1


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _train_transform(config: dict):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(int(config["image_size"]), scale=tuple(config.get("crop_scale", [0.3, 1.0]))),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _eval_transform(config: dict):
    from torchvision import transforms

    image_size = int(config["image_size"])
    return transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class ImageNet100(Dataset):
    """clane9/imagenet-100 parquet shards (train: 126,689 images, validation: 5,000; 100 classes)."""

    def __init__(self, root: str | Path, split: str, transform, cache_dir: str | None = None) -> None:
        from datasets import load_dataset

        self.data = load_dataset("parquet", data_dir=str(root), split=split, cache_dir=cache_dir)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[int(index)]
        return self.transform(row["image"].convert("RGB")), row["label"]


def make_dataset(config: dict, train: bool = True, split: str | None = None):
    name = config["name"]
    image_size = int(config["image_size"])
    if name == "synthetic":
        return FixedSyntheticImages(int(config.get("synthetic_count", 32)), image_size, int(config.get("seed", 0)))
    augment = train and split is None
    transform = _train_transform(config) if augment else _eval_transform(config)
    if name == "stl10":
        from torchvision import datasets

        dataset_split = "unlabeled" if augment else split or "test"
        return datasets.STL10(root=str(Path(config["root"])), split=dataset_split, download=True, transform=transform)
    if name == "imagenet100":
        # "test" maps to the validation shard so kNN evaluation works unchanged.
        dataset_split = "train" if augment else {"test": "validation"}.get(split or "test", split)
        return ImageNet100(config["root"], dataset_split, transform, config.get("cache_dir"))
    raise ValueError(f"unknown dataset: {name}")


def make_loader(dataset, config: dict, epoch: int = 0, collate_fn=None):
    generator = torch.Generator().manual_seed(int(config.get("seed", 0)) + epoch)
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=bool(config.get("pin_memory", True)),
        persistent_workers=int(config.get("num_workers", 0)) > 0,
        generator=generator,
        collate_fn=collate_fn,
    )
