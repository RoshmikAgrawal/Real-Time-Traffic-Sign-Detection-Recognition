# ============================================================================
# data_loader.py — Dataset Loading, Preprocessing & Augmentation
# ============================================================================
"""
Handles all data-related operations for the GTSRB traffic sign dataset:
  • Loading images and labels from directory structure or CSV
  • Resizing and normalizing images
  • Splitting into train / validation / test sets
  • Creating data augmentation generators for training
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.config import (
    DATASET_DIR, TRAIN_CSV, TEST_CSV,
    IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES, VALIDATION_SPLIT,
    ROTATION_RANGE, WIDTH_SHIFT_RANGE, HEIGHT_SHIFT_RANGE,
    ZOOM_RANGE, SHEAR_RANGE, BRIGHTNESS_RANGE,
)


# ──────────────────────────────────────────────────────────────────────
# Image Loading Helpers
# ──────────────────────────────────────────────────────────────────────

def load_image(image_path: str) -> np.ndarray:
    """
    Load a single image, resize to (IMG_HEIGHT, IMG_WIDTH), and return
    as a NumPy array with pixel values in [0, 255].
    """
    img = Image.open(image_path)
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    return np.array(img)


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values from [0, 255] to [0, 1] for neural network input.
    """
    return image.astype(np.float32) / 255.0


# ──────────────────────────────────────────────────────────────────────
# Dataset Loaders
# ──────────────────────────────────────────────────────────────────────

def load_train_data_from_directory() -> tuple:
    """
    Load training images directly from the GTSRB directory structure:
        dataset/Train/0/  dataset/Train/1/  ...  dataset/Train/42/

    Returns:
        images  (np.ndarray): Array of shape (N, IMG_HEIGHT, IMG_WIDTH, 3)
        labels  (np.ndarray): Array of integer class labels, shape (N,)
    """
    images, labels = [], []
    train_dir = os.path.join(DATASET_DIR, "Train")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(
            f"Training directory not found at '{train_dir}'. "
            "Please download the GTSRB dataset first. "
            "See dataset/README.md for instructions."
        )

    print("[INFO] Loading training images from directory structure...")
    for class_id in range(NUM_CLASSES):
        class_dir = os.path.join(train_dir, str(class_id))
        if not os.path.isdir(class_dir):
            print(f"  [WARN] Class directory missing: {class_dir}")
            continue

        class_images = os.listdir(class_dir)
        for img_name in class_images:
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.ppm')):
                continue
            try:
                img_path = os.path.join(class_dir, img_name)
                img = load_image(img_path)
                images.append(img)
                labels.append(class_id)
            except Exception as e:
                print(f"  [WARN] Failed to load {img_name}: {e}")

    images = np.array(images)
    labels = np.array(labels)
    print(f"[INFO] Loaded {len(images)} training images across {NUM_CLASSES} classes.")
    return images, labels


def load_test_data_from_csv() -> tuple:
    """
    Load test images using the Test.csv file that maps image paths to labels.

    Returns:
        images  (np.ndarray): Array of shape (N, IMG_HEIGHT, IMG_WIDTH, 3)
        labels  (np.ndarray): Array of integer class labels, shape (N,)
    """
    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(
            f"Test CSV not found at '{TEST_CSV}'. "
            "Please download the GTSRB dataset first. "
            "See dataset/README.md for instructions."
        )

    print("[INFO] Loading test images from CSV...")
    test_df = pd.read_csv(TEST_CSV)
    images, labels = [], []

    for _, row in test_df.iterrows():
        img_path = os.path.join(DATASET_DIR, row['Path'])
        try:
            img = load_image(img_path)
            images.append(img)
            labels.append(row['ClassId'])
        except Exception as e:
            print(f"  [WARN] Failed to load {row['Path']}: {e}")

    images = np.array(images)
    labels = np.array(labels)
    print(f"[INFO] Loaded {len(images)} test images.")
    return images, labels


# ──────────────────────────────────────────────────────────────────────
# Data Splitting & Preparation
# ──────────────────────────────────────────────────────────────────────

def prepare_data() -> dict:
    """
    Full data preparation pipeline:
      1. Load raw training images
      2. Normalize pixel values
      3. One-hot encode labels
      4. Split into training and validation sets
      5. Load test data separately

    Returns:
        dict with keys: X_train, y_train, X_val, y_val, X_test, y_test
    """
    # Load training data
    images, labels = load_train_data_from_directory()

    # Normalize images to [0, 1]
    images = preprocess_image(images)

    # One-hot encode labels
    labels_one_hot = to_categorical(labels, NUM_CLASSES)

    # Train / Validation split
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels_one_hot,
        test_size=VALIDATION_SPLIT,
        random_state=42,
        stratify=labels  # Preserve class distribution
    )

    print(f"[INFO] Training set   : {X_train.shape[0]} samples")
    print(f"[INFO] Validation set : {X_val.shape[0]} samples")

    # Load test data
    try:
        X_test, y_test_raw = load_test_data_from_csv()
        X_test = preprocess_image(X_test)
        y_test = to_categorical(y_test_raw, NUM_CLASSES)
        print(f"[INFO] Test set       : {X_test.shape[0]} samples")
    except FileNotFoundError:
        print("[WARN] Test CSV not found — skipping test set loading.")
        X_test, y_test = None, None

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val":   X_val,
        "y_val":   y_val,
        "X_test":  X_test,
        "y_test":  y_test,
    }


# ──────────────────────────────────────────────────────────────────────
# Data Augmentation Generator
# ──────────────────────────────────────────────────────────────────────

def get_train_generator(X_train: np.ndarray, y_train: np.ndarray,
                        batch_size: int) -> ImageDataGenerator:
    """
    Create an augmented data generator for the training set.

    Augmentations applied:
      • Random rotation (±15°)
      • Horizontal & vertical shifts
      • Zoom & shear transformations
      • Brightness adjustment

    These simulate real-world variations in traffic sign appearance
    (different angles, distances, lighting conditions).

    Args:
        X_train:    Training images array
        y_train:    One-hot encoded labels
        batch_size: Number of samples per batch

    Returns:
        A Keras ImageDataGenerator flow
    """
    datagen = ImageDataGenerator(
        rotation_range=ROTATION_RANGE,
        width_shift_range=WIDTH_SHIFT_RANGE,
        height_shift_range=HEIGHT_SHIFT_RANGE,
        zoom_range=ZOOM_RANGE,
        shear_range=SHEAR_RANGE,
        brightness_range=BRIGHTNESS_RANGE,
        fill_mode='nearest',
    )

    return datagen.flow(X_train, y_train, batch_size=batch_size)


# ──────────────────────────────────────────────────────────────────────
# Quick Test
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data = prepare_data()
    print("\n✅ Data loading complete!")
    print(f"   X_train shape : {data['X_train'].shape}")
    print(f"   y_train shape : {data['y_train'].shape}")
    if data["X_test"] is not None:
        print(f"   X_test shape  : {data['X_test'].shape}")
