import yaml
import os
from pathlib import Path

def load_config(config_path="config.yaml"):
    """Loads the YAML config and resolves paths to absolute paths."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Resolve paths relative to the current working directory
    if "paths" in config:
        for key, val in config["paths"].items():
            if val is not None and isinstance(val, str):
                # Expand ~ for Linux/Mac and resolve to absolute paths
                resolved_path = Path(val).expanduser().resolve()
                config["paths"][key] = str(resolved_path)

    return config