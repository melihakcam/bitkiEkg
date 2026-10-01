"""Proje ayarlarını ve klasör yollarını yükler."""

from pathlib import Path
import random

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_path(name: str, cfg: dict | None = None) -> Path:
    """configs/config.yaml içindeki `paths` anahtarından mutlak yol döndürür."""
    cfg = cfg or load_config()
    path = PROJECT_ROOT / cfg["paths"][name]
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int | None = None) -> int:
    """Tekrarlanabilirlik için rastgelelik tohumunu sabitler."""
    seed = load_config()["seed"] if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    return seed
