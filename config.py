from pathlib import Path
import torch

# General Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directories
ARTIFACTS_DIR = Path("artifacts")
# Correctly set DATA_DIR to point to the subfolder containing the class directories
DATA_DIR = ARTIFACTS_DIR / "via-cervix" 
RESULTS_DIR = ARTIFACTS_DIR / "training_runs"

# Class names and mapping
CLASS_NAMES = ["Negative", "Positive", "Suspicious cancer"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

# Model Configuration
MODEL_NAME = "google/vit-base-patch16-224"
NUM_LABELS = len(CLASS_NAMES)
BASE_MODEL_PATH = ARTIFACTS_DIR / "base_model"
TRAINED_MODEL_PATH = RESULTS_DIR / "best_model.pth"

# Training Hyperparameters
SEED = 42
EPOCHS = 10
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
CANCER_PENALTY = 15.0
NUM_SPLITS = 5
BATCH_SIZE = 16

# Data Ingestion (assuming this part is for downloading, which might be handled elsewhere)
FILE_ID = "1lFvuTpzdfSAckyjtHZzE2HWqsH25sa1q"
ZIP_PATH = ARTIFACTS_DIR / "via-cervix.zip"
EXTRACT_DIR = ARTIFACTS_DIR / "via-cervix"