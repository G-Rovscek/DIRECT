import yaml
import glob
import os
from pprint import pprint


def deep_update(original, updates):
    for k, v in updates.items():
        if isinstance(v, dict) and k in original:
            deep_update(original[k], v)
        else:
            original[k] = v
    return original


def load_first_yaml(model_dir):
    yaml_files = glob.glob(os.path.join(model_dir, "*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No .yaml config found in {model_dir}")

    # Pick the first one (sorted ensures deterministic choice)
    yaml_files.sort()
    config_path = yaml_files[0]

    cfg = load_config(config_path)

    return cfg


def load_config(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    print(f"{config_path} - Loaded configuration: ")
    pprint(cfg)
    return cfg


def save_config(cfg, config_path):
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)