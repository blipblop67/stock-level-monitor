import os
from pathlib import Path

import yaml


def load_config(path):
    path = Path(path).resolve()
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_base"] = path.parent
    cfg["_trays_file"] = resolve(cfg, cfg.get("trays_file", "trays.yaml"))
    cfg["trays"] = load_trays(cfg["_trays_file"])
    return cfg


def load_trays(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("trays", []) or []


def save_trays(path, trays):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Written by roi_tool.py. Polygon points are normalised (0-1).\n")
        yaml.safe_dump({"trays": trays}, f, sort_keys=False, allow_unicode=True)


def resolve(cfg, p):
    """Resolve a path from the config relative to the config file."""
    p = Path(p)
    return p if p.is_absolute() else cfg["_base"] / p


def secret(env_name, default=""):
    return os.environ.get(env_name or "", default)
