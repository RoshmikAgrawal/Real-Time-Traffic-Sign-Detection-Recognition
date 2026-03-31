# ============================================================================
# train.py — Training Pipeline for Traffic Sign Classifier
# ============================================================================
"""
Orchestrates the full model training workflow:
  1. Load and prepare data (with augmentation)
  2. Build the CNN model
  3. Configure callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
  4. Train the model
  5. Save the best model and training history
  6. Plot training curves (accuracy & loss)
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)

from src.config import (
    BATCH_SIZE, EPOCHS, MODEL_SAVE_PATH, HISTORY_PATH,
    OUTPUT_DIR, EARLY_STOPPING_PATIENCE,
    REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR,
    ensure_directories,
)
from src.data_loader import prepare_data, get_train_generator
from src.model import build_traffic_sign_model


def get_callbacks() -> list:
    """
    Configure training callbacks:

    • EarlyStopping: Halt training when validation accuracy plateaus
      to prevent overfitting and save time.

    • ReduceLROnPlateau: Lower the learning rate when validation loss
      stops improving, allowing finer weight updates.

    • ModelCheckpoint: Save only the best model (based on val_accuracy)
      so we always keep the top-performing weights.

    Returns:
        List of Keras callback instances.
    """
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
        ),
    ]
    return callbacks


def plot_training_history(history) -> None:
    """
    Generate and save training/validation accuracy & loss curves.

    These plots help diagnose:
      • Overfitting (training acc ↑↑ but val acc ↓)
      • Underfitting (both accuracies remain low)
      • Good convergence (both curves plateau together)

    Args:
        history: Keras History object from model.fit()
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Accuracy Plot ────────────────────────────────────────────────
    axes[0].plot(history.history['accuracy'], label='Train Accuracy',
                 linewidth=2, color='#2196F3')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy',
                 linewidth=2, color='#FF5722')
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])

    # ── Loss Plot ────────────────────────────────────────────────────
    axes[1].plot(history.history['loss'], label='Train Loss',
                 linewidth=2, color='#2196F3')
    axes[1].plot(history.history['val_loss'], label='Validation Loss',
                 linewidth=2, color='#FF5722')
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "training_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Training curves saved to: {plot_path}")


def train_model() -> None:
    """
    Execute the full training pipeline:
      1. Ensure directories exist
      2. Load & preprocess data
      3. Build the CNN model
      4. Train with data augmentation
      5. Save model and history
      6. Plot training curves
    """
    ensure_directories()

    # ── Step 1: Load Data ────────────────────────────────────────────
    print("=" * 60)
    print("  STEP 1: Loading and Preparing Data")
    print("=" * 60)
    data = prepare_data()
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val     = data["X_val"], data["y_val"]

    # ── Step 2: Build Model ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 2: Building CNN Model")
    print("=" * 60)
    model = build_traffic_sign_model()
    model.summary()

    # ── Step 3: Set Up Data Augmentation ─────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 3: Configuring Data Augmentation")
    print("=" * 60)
    train_generator = get_train_generator(X_train, y_train, BATCH_SIZE)
    steps_per_epoch = len(X_train) // BATCH_SIZE
    print(f"[INFO] Steps per epoch: {steps_per_epoch}")
    print(f"[INFO] Batch size: {BATCH_SIZE}")

    # ── Step 4: Train ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 4: Training the Model")
    print("=" * 60)
    callbacks = get_callbacks()

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    # ── Step 5: Save Training History ────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 5: Saving Model and History")
    print("=" * 60)
    with open(HISTORY_PATH, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"[INFO] Training history saved to: {HISTORY_PATH}")
    print(f"[INFO] Best model saved to: {MODEL_SAVE_PATH}")

    # ── Step 6: Plot Training Curves ─────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 6: Generating Training Plots")
    print("=" * 60)
    plot_training_history(history)

    # ── Summary ──────────────────────────────────────────────────────
    best_val_acc = max(history.history['val_accuracy'])
    best_epoch = np.argmax(history.history['val_accuracy']) + 1
    print("\n" + "=" * 60)
    print("  ✅ TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best Validation Accuracy : {best_val_acc:.4f}")
    print(f"  Achieved at Epoch        : {best_epoch}")
    print(f"  Total Epochs Run         : {len(history.history['accuracy'])}")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────────────
# Run training if executed directly
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_model()
