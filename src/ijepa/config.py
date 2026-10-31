from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Config:
    raw: dict

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as handle:
            return cls(yaml.safe_load(handle))

    def __getitem__(self, key):
        return self.raw[key]
