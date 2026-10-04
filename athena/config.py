"""Load Athena configuration from configs/*.yaml."""

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


@lru_cache(maxsize=None)
def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def data_path(key: str) -> Path:
    """Absolute path for an entry under `paths:` in the config."""
    return ROOT / load_config()["paths"][key]
