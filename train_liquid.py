"""
Training script for Liquid Chart Generator.

Usage:
    # On cloud (Mug-Diffusion at /notebooks/Mug-Diffusion):
    MUG_DIFFUSION_PATH=/notebooks/Mug-Diffusion python train_liquid.py fit --config configs/train_liquid.yaml

    # Locally (Mug-Diffusion at /Volumes/XelesteSSD/Mug-Diffusion):
    MUG_DIFFUSION_PATH=/Volumes/XelesteSSD/Mug-Diffusion python train_liquid.py fit --config configs/train_liquid.yaml

Imports data pipeline from Mug-Diffusion project (mai.data.dataset, mai.firststage.losses).
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys

# Add Mug-Diffusion to path for data loading and loss functions
MUG_DIFFUSION_PATH = os.environ.get("MUG_DIFFUSION_PATH", "/Volumes/XelesteSSD/Mug-Diffusion")
if os.path.isdir(MUG_DIFFUSION_PATH):
    sys.path.insert(0, MUG_DIFFUSION_PATH)
    print(f"[liquid] Added Mug-Diffusion path: {MUG_DIFFUSION_PATH}")
else:
    raise RuntimeError(
        f"Mug-Diffusion path not found: {MUG_DIFFUSION_PATH}\n"
        f"Set MUG_DIFFUSION_PATH env var to your Mug-Diffusion project root."
    )

import torch
torch.set_float32_matmul_precision('medium')

from pytorch_lightning.cli import LightningCLI

# Import DataModuleFromConfig from Mug-Diffusion's main.py
# This handles instantiation of mai.data.dataset.MaimaiTrainDataset etc.
from main import DataModuleFromConfig

from models.liquid_seq2seq import LiquidChartGenerator


def cli_main():
    cli = LightningCLI(
        model_class=LiquidChartGenerator,
        datamodule_class=DataModuleFromConfig,
        save_config_kwargs={"overwrite": True}
    )


if __name__ == '__main__':
    cli_main()
