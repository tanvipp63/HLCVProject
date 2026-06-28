from pathlib import Path

import yaml


def load_config(config_path: str = "configs/paths.yaml") -> dict:
    config_path = Path(config_path).expanduser()

    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parents[2] / config_path

    if not config_path.exists():
        alt_path = config_path.with_suffix(".yml")
        if alt_path.exists():
            config_path = alt_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)