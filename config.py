"""
config.py
=========

Central configuration file for the Waste Classification project.
"""

import os
import torch

# -----------------------------------------------------------
# Project Paths
# -----------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# TrashNet Dataset
TRASHNET_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "trashnet"
)

# RealWaste Dataset
REALWASTE_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "Realtime",
    "Realtime",
    "realwaste-main",
    "RealWaste"
)

# Output folders
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------------------------------------
# Dataset Classes
# -----------------------------------------------------------

TRASHNET_CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

REALWASTE_CLASSES = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Miscellaneous Trash",
    "Paper",
    "Plastic",
    "Textile Trash",
    "Vegetation"
]

# Mapping from RealWaste classes to TrashNet classes
REALWASTE_TO_TRASHNET = {
    "Cardboard": "cardboard",
    "Glass": "glass",
    "Metal": "metal",
    "Paper": "paper",
    "Plastic": "plastic",
    "Miscellaneous Trash": "trash",
}

# -----------------------------------------------------------
# Training Configuration
# -----------------------------------------------------------

IMG_SIZE = 224

BATCH_SIZE = 32

NUM_EPOCHS = 20

LR = 3e-4

WEIGHT_DECAY = 1e-4

TRAIN_SPLIT = 0.70

VAL_SPLIT = 0.15

TEST_SPLIT = 0.15

RANDOM_SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# -----------------------------------------------------------
# CNN Backbones
# -----------------------------------------------------------

BACKBONES = [
    "resnet50",
    "efficientnet_b0",
    "mobilenet_v3_large"
]

# -----------------------------------------------------------
# Domain Generalization
# -----------------------------------------------------------

DG_FINETUNE_FRACTION = 0.20