# Given a config file, calculate the stats of the dataset

import os
import pickle

import hydra
import torch
from omegaconf import OmegaConf
from pathlib import Path

def get_stats(cfg):
    stats = {}
    for data_dir in cfg.all_data_directories:
        dset = hydra.utils.instantiate(cfg.dataset, data_dir=data_dir)
        dset_stats = dset.get_stats()
        for key, value in dset_stats.items():
            if key in stats:
                stats[key].append(torch.stack(value, dim=0))
            else:
                stats[key] = [torch.stack(value, dim=0)]

    for key in stats.keys():
        stats[key] = torch.mean(torch.stack(stats[key], dim=0), dim=0)

    print(f"stats: {stats}")
    return stats


def load_config(training_path):
    cfg = OmegaConf.load(os.path.join(training_path, ".hydra/config.yaml"))
    return cfg

def collect_training_paths(date_dir: str | Path):
    date_dir = Path(date_dir)
    if not date_dir.is_dir():
        raise FileNotFoundError(f"{date_dir} not found")
    runs = []
    for root, dirs, files in os.walk(date_dir):
        if 'models' in dirs:          
            runs.append(Path(root))
    runs.sort(key=lambda p: p.stat().st_mtime)
    return runs

if __name__ == "__main__":
    BASE = "/home/jolia/vr-hand-tracking/Franka-Teach/RUKA/checkpoints/control-trainings/out"
    DATE = "2025.08.17"
    training_paths = collect_training_paths(Path(BASE) / DATE)

    for training_path in training_paths:
        cfg = load_config(training_path)
        stats = get_stats(cfg)
        with open(os.path.join(training_path, "dataset_stats.pkl"), "wb") as f:
            pickle.dump(stats, f)
