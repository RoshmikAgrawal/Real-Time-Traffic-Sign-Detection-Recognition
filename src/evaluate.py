# ============================================================================
# evaluate.py — Model Evaluation & Performance Metrics
# ============================================================================
"""
Comprehensive evaluation of the trained traffic sign classifier:
  • Overall test accuracy
  • Confusion matrix (with heatmap visualization)
  • Per-class classification report (precision, recall, F1-score)
  • Misclassified samples visualization
  • Per-class accuracy breakdown
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from tensorflow.keras.models import load_model

from src.config import (
    MODEL_SAVE_PATH, OUTPUT_DIR, NUM_CLASSES,
    CLASS_NAMES, get_class_name, ensure_directories,
)
from src.data_loader import prepare_data


def load_trained_model():
    """
    Load the best saved model from disk.

    Returns:
        Compiled Keras model ready for inference.
    """
    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"No trained model found at '{MODEL_SAVE_PATH}'. "
            "Please run training first: python main.py train"
        )
    print(f"[INFO] Loading model from: {MODEL_SAVE_PATH}")
    model = load_model(MODEL_SAVE_PATH)
    return model


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Generate and save a heatmap of the confusion matrix.

    The confusion matrix reveals:
      • Which classes are frequently confused with each other
      • Class-specific strengths and weaknesses of the model
      • Diagonal dominance indicates good performance

    Args:
        y_true: Ground truth class indices.
        y_pred: Predicted class indices.
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm, annot=False, fmt='d', cmap='Blues',
        xticklabels=range(NUM_CLASSES),
        yticklabels=range(NUM_CLASSES),
    )
    plt.title('Confusion Matrix — Traffic Sign Classification', fontsize=16)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Confusion matrix saved to: {save_path}")


def plot_per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Bar chart showing accuracy for each of the 43 sign classes.

    This helps identify underperforming classes that may need:
      • More training samples (data imbalance)
      • Targeted augmentation
      • Feature engineering

    Args:
        y_true: Ground truth class indices.
        y_pred: Predicted class indices.
    """
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    plt.figure(figsize=(16, 6))
    colors = ['#4CAF50' if acc >= 0.9 else '#FF9800' if acc >= 0.7 else '#F44336'
              for acc in per_class_acc]
    plt.bar(range(NUM_CLASSES), per_class_acc, color=colors, edgecolor='black',
            linewidth=0.5)
    plt.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='90% threshold')
    plt.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='70% threshold')
    plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Class ID')
    plt.ylabel('Accuracy')
    plt.xticks(range(NUM_CLASSES), fontsize=7)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "per_class_accuracy.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Per-class accuracy chart saved to: {save_path}")


def plot_misclassified_samples(X_test: np.ndarray, y_true: np.ndarray,
                                y_pred: np.ndarray, num_samples: int = 16) -> None:
    """
    Display a grid of misclassified images with their true vs. predicted labels.

    This visual inspection can reveal:
      • Ambiguous or damaged images in the dataset
      • Signs that look similar across classes
      • Systematic errors the model makes

    Args:
        X_test:      Test images (normalized 0–1).
        y_true:      Ground truth class indices.
        y_pred:      Predicted class indices.
        num_samples: Number of misclassified examples to display.
    """
    misclassified_idx = np.where(y_true != y_pred)[0]

    if len(misclassified_idx) == 0:
        print("[INFO] No misclassified samples — perfect accuracy! 🎉")
        return

    # Select random subset of misclassified images
    sample_idx = np.random.choice(
        misclassified_idx,
        size=min(num_samples, len(misclassified_idx)),
        replace=False,
    )

    cols = 4
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.5 * rows))
    axes = axes.flatten()

    for i, idx in enumerate(sample_idx):
        axes[i].imshow(X_test[idx])
        axes[i].set_title(
            f"True: {get_class_name(y_true[idx])}\n"
            f"Pred: {get_class_name(y_pred[idx])}",
            fontsize=7, color='red',
        )
        axes[i].axis('off')

    # Hide unused subplots
    for j in range(len(sample_idx), len(axes)):
        axes[j].axis('off')

    plt.suptitle('Misclassified Samples', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "misclassified_samples.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Misclassified samples saved to: {save_path}")


def evaluate_model() -> None:
    """
    Run the full evaluation pipeline:
      1. Load the trained model
      2. Load test data
      3. Generate predictions
      4. Compute and display metrics
      5. Create visualizations
    """
    ensure_directories()

    # ── Load Model ───────────────────────────────────────────────────
    print("=" * 60)
    print("  MODEL EVALUATION")
    print("=" * 60)
    model = load_trained_model()

    # ── Load Test Data ───────────────────────────────────────────────
    print("\n[INFO] Loading test data...")
    data = prepare_data()
    X_test, y_test = data["X_test"], data["y_test"]

    if X_test is None:
        print("[ERROR] Test data not available. Cannot evaluate.")
        return

    # ── Generate Predictions ─────────────────────────────────────────
    print("[INFO] Generating predictions on test set...")
    y_pred_proba = model.predict(X_test, batch_size=64, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # ── Overall Accuracy ─────────────────────────────────────────────
    overall_acc = accuracy_score(y_true, y_pred)
    print(f"\n{'='*60}")
    print(f"  📊 Overall Test Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
    print(f"{'='*60}")

    # ── Classification Report ────────────────────────────────────────
    print("\n📋 Classification Report:")
    print("-" * 60)
    target_names = [get_class_name(i) for i in range(NUM_CLASSES)]
    report = classification_report(y_true, y_pred, target_names=target_names)
    print(report)

    # Save classification report to file
    report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"Overall Test Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)\n\n")
        f.write(report)
    print(f"[INFO] Classification report saved to: {report_path}")

    # ── Visualizations ───────────────────────────────────────────────
    print("\n[INFO] Generating evaluation plots...")
    plot_confusion_matrix(y_true, y_pred)
    plot_per_class_accuracy(y_true, y_pred)
    plot_misclassified_samples(X_test, y_true, y_pred)

    print("\n✅ Evaluation complete! Check the 'outputs/' directory for results.")


# ──────────────────────────────────────────────────────────────────────
# Run evaluation if executed directly
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    evaluate_model()
