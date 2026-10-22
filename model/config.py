
from pathlib import Path
import torch

# Project directories
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "PlantVillage"
MODEL_DIR = PROJECT_DIR / "model"
SAVE_DIR = PROJECT_DIR / "saved_models"

# Dataset settings
IMAGE_SIZE = 224
BATCH_SIZE = 32
TRAIN_SPLIT = 0.80
VAL_SPLIT = 0.10
TEST_SPLIT = 0.10
SEED = 42

# Model training settings
EPOCHS = 30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 7

# Learning rate scheduler
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 2

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Saved files
BEST_MODEL_PATH = PROJECT_DIR / "saved_models" / "best_model.pth"
CLASS_NAMES_PATH = PROJECT_DIR / "saved_models" / "class_names.json"

# Create the model output directory
SAVE_DIR.mkdir(parents=True, exist_ok=True)
