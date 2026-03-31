# ============================================================================
# config.py — Central Configuration for Traffic Sign Recognition
# ============================================================================
"""
Contains all hyperparameters, file paths, and class label mappings
used throughout the project. Modify values here to tune the model
without touching other source files.
"""

import os

# ──────────────────────────────────────────────────────────────────────
# Path Configuration
# ──────────────────────────────────────────────────────────────────────

# Base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset paths
DATASET_DIR      = os.path.join(BASE_DIR, "dataset")
TRAIN_CSV        = os.path.join(DATASET_DIR, "Train.csv")
TEST_CSV         = os.path.join(DATASET_DIR, "Test.csv")

# Model save path
MODEL_DIR        = os.path.join(BASE_DIR, "models")
MODEL_SAVE_PATH  = os.path.join(MODEL_DIR, "traffic_sign_model.keras")
HISTORY_PATH     = os.path.join(MODEL_DIR, "training_history.pkl")

# Output directory for plots and reports
OUTPUT_DIR       = os.path.join(BASE_DIR, "outputs")

# ──────────────────────────────────────────────────────────────────────
# Image & Data Configuration
# ──────────────────────────────────────────────────────────────────────

IMG_HEIGHT       = 32          # Resize height for input images
IMG_WIDTH        = 32          # Resize width for input images
IMG_CHANNELS     = 3           # Number of color channels (RGB)
INPUT_SHAPE      = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# ──────────────────────────────────────────────────────────────────────
# Training Hyperparameters
# ──────────────────────────────────────────────────────────────────────

NUM_CLASSES      = 43          # GTSRB has 43 classes of traffic signs
BATCH_SIZE       = 64          # Mini-batch size for training
EPOCHS           = 30          # Maximum number of training epochs
LEARNING_RATE    = 0.001       # Initial learning rate (Adam optimizer)
VALIDATION_SPLIT = 0.2         # Fraction of training data used for validation

# ──────────────────────────────────────────────────────────────────────
# Data Augmentation Parameters
# ──────────────────────────────────────────────────────────────────────

ROTATION_RANGE     = 15        # Random rotation ±15 degrees
WIDTH_SHIFT_RANGE  = 0.1       # Horizontal shift fraction
HEIGHT_SHIFT_RANGE = 0.1       # Vertical shift fraction
ZOOM_RANGE         = 0.2       # Random zoom range
SHEAR_RANGE        = 0.1       # Shear intensity
BRIGHTNESS_RANGE   = (0.8, 1.2)  # Brightness adjustment range

# ──────────────────────────────────────────────────────────────────────
# Callbacks Configuration
# ──────────────────────────────────────────────────────────────────────

EARLY_STOPPING_PATIENCE = 7    # Stop if no improvement for N epochs
REDUCE_LR_PATIENCE      = 3   # Reduce LR if no improvement for N epochs
REDUCE_LR_FACTOR        = 0.5 # Multiply LR by this factor on plateau

# ──────────────────────────────────────────────────────────────────────
# GTSRB Class Labels (43 Traffic Sign Categories)
# ──────────────────────────────────────────────────────────────────────

CLASS_NAMES = {
    0:  "Speed limit (20km/h)",
    1:  "Speed limit (30km/h)",
    2:  "Speed limit (50km/h)",
    3:  "Speed limit (60km/h)",
    4:  "Speed limit (70km/h)",
    5:  "Speed limit (80km/h)",
    6:  "End of speed limit (80km/h)",
    7:  "Speed limit (100km/h)",
    8:  "Speed limit (120km/h)",
    9:  "No passing",
    10: "No passing for vehicles over 3.5t",
    11: "Right-of-way at next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5t prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5t",
}

# ──────────────────────────────────────────────────────────────────────
# Convenience function
# ──────────────────────────────────────────────────────────────────────

def get_class_name(class_id: int) -> str:
    """Return the human-readable name for a given class ID."""
    return CLASS_NAMES.get(class_id, f"Unknown class ({class_id})")


def ensure_directories():
    """Create necessary directories if they don't exist."""
    for directory in [MODEL_DIR, OUTPUT_DIR, DATASET_DIR]:
        os.makedirs(directory, exist_ok=True)
