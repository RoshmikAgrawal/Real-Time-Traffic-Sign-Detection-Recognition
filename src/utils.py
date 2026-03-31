# ============================================================================
# utils.py — Utility Functions for Visualization and Helpers
# ============================================================================
"""
Collection of helper functions used across the project:
  • Dataset distribution visualization
  • Sample images display
  • Image preprocessing utilities
  • Data augmentation preview
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

from src.config import NUM_CLASSES, CLASS_NAMES, OUTPUT_DIR, ensure_directories


def plot_class_distribution(labels: np.ndarray, title: str = "Class Distribution",
                            filename: str = "class_distribution.png") -> None:
    """
    Plot a bar chart showing the number of samples per class.

    This reveals class imbalance — a common issue in traffic sign datasets
    where some signs (e.g., speed limits) appear far more often than others
    (e.g., end-of-speed-limit signs), which can bias the model.

    Args:
        labels:   Array of integer class labels.
        title:    Plot title.
        filename: Output filename.
    """
    ensure_directories()

    class_counts = Counter(labels)
    classes = sorted(class_counts.keys())
    counts = [class_counts[c] for c in classes]

    plt.figure(figsize=(16, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))
    plt.bar(classes, counts, color=colors, edgecolor='black', linewidth=0.3)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Class ID')
    plt.ylabel('Number of Samples')
    plt.xticks(classes, fontsize=7)
    plt.grid(axis='y', alpha=0.3)

    # Annotate min and max classes
    min_class = min(class_counts, key=class_counts.get)
    max_class = max(class_counts, key=class_counts.get)
    plt.annotate(f'Min: Class {min_class}\n({class_counts[min_class]} samples)',
                 xy=(min_class, class_counts[min_class]),
                 fontsize=8, color='red', fontweight='bold')
    plt.annotate(f'Max: Class {max_class}\n({class_counts[max_class]} samples)',
                 xy=(max_class, class_counts[max_class]),
                 fontsize=8, color='blue', fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Class distribution plot saved to: {save_path}")


def display_sample_images(images: np.ndarray, labels: np.ndarray,
                          num_per_class: int = 1,
                          filename: str = "sample_images.png") -> None:
    """
    Display a grid of sample images, one per class.

    Useful for visually inspecting the dataset to understand what
    each traffic sign category looks like.

    Args:
        images:        Array of images.
        labels:        Array of corresponding class labels.
        num_per_class: Number of samples to show per class.
        filename:      Output filename.
    """
    ensure_directories()

    cols = 9
    rows = (NUM_CLASSES + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 2.2 * rows))
    axes = axes.flatten()

    for class_id in range(NUM_CLASSES):
        class_mask = (labels == class_id)
        class_indices = np.where(class_mask)[0]

        if len(class_indices) > 0:
            idx = class_indices[0]
            img = images[idx]
            # Handle both normalized (0-1) and raw (0-255) images
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            axes[class_id].imshow(img)
            axes[class_id].set_title(f"[{class_id}]", fontsize=7)
        else:
            axes[class_id].set_title(f"[{class_id}] N/A", fontsize=7, color='red')

        axes[class_id].axis('off')

    # Hide unused subplots
    for j in range(NUM_CLASSES, len(axes)):
        axes[j].axis('off')

    plt.suptitle('Sample Images for Each Traffic Sign Class',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Sample images saved to: {save_path}")


def display_augmented_samples(images: np.ndarray, labels: np.ndarray,
                               num_samples: int = 8,
                               filename: str = "augmented_samples.png") -> None:
    """
    Show original images alongside their augmented versions.

    This demonstrates the effect of data augmentation and helps verify
    that augmentations are realistic and not too extreme.

    Args:
        images:      Original training images (normalized 0–1).
        labels:      Corresponding class labels.
        num_samples: Number of images to augment and display.
        filename:    Output filename.
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from src.config import (
        ROTATION_RANGE, WIDTH_SHIFT_RANGE, HEIGHT_SHIFT_RANGE,
        ZOOM_RANGE, SHEAR_RANGE, BRIGHTNESS_RANGE,
    )

    ensure_directories()

    datagen = ImageDataGenerator(
        rotation_range=ROTATION_RANGE,
        width_shift_range=WIDTH_SHIFT_RANGE,
        height_shift_range=HEIGHT_SHIFT_RANGE,
        zoom_range=ZOOM_RANGE,
        shear_range=SHEAR_RANGE,
        brightness_range=BRIGHTNESS_RANGE,
    )

    fig, axes = plt.subplots(num_samples, 5, figsize=(12, 2.5 * num_samples))

    random_indices = np.random.choice(len(images), num_samples, replace=False)

    for row, idx in enumerate(random_indices):
        img = images[idx]
        label = labels[idx] if isinstance(labels[idx], (int, np.integer)) else np.argmax(labels[idx])

        # Original image
        display = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img
        axes[row, 0].imshow(display)
        axes[row, 0].set_title(f"Original [{label}]", fontsize=8)
        axes[row, 0].axis('off')

        # 4 augmented versions
        img_batch = np.expand_dims(img, axis=0)
        aug_iter = datagen.flow(img_batch, batch_size=1)
        for col in range(1, 5):
            aug_img = next(aug_iter)[0]
            aug_display = np.clip(aug_img, 0, 1)
            axes[row, col].imshow(aug_display)
            axes[row, col].set_title(f"Aug {col}", fontsize=8)
            axes[row, col].axis('off')

    plt.suptitle('Data Augmentation Preview', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Augmented samples saved to: {save_path}")


def print_dataset_summary(labels: np.ndarray) -> None:
    """
    Print a formatted table of dataset statistics.

    Args:
        labels: Array of integer class labels.
    """
    class_counts = Counter(labels)
    total = len(labels)

    print(f"\n{'='*65}")
    print(f"  📊 DATASET SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Class ID':<10}{'Class Name':<40}{'Count':<8}{'%':<8}")
    print(f"  {'-'*63}")

    for class_id in range(NUM_CLASSES):
        count = class_counts.get(class_id, 0)
        pct = (count / total * 100) if total > 0 else 0
        name = CLASS_NAMES.get(class_id, "Unknown")
        print(f"  {class_id:<10}{name:<40}{count:<8}{pct:>5.1f}%")

    print(f"  {'-'*63}")
    print(f"  {'TOTAL':<50}{total:<8}{'100.0%':>6}")
    print(f"{'='*65}")
    print(f"  Min class: {min(class_counts, key=class_counts.get)} "
          f"({class_counts[min(class_counts, key=class_counts.get)]} samples)")
    print(f"  Max class: {max(class_counts, key=class_counts.get)} "
          f"({class_counts[max(class_counts, key=class_counts.get)]} samples)")
    print(f"  Imbalance ratio: "
          f"{class_counts[max(class_counts, key=class_counts.get)] / class_counts[min(class_counts, key=class_counts.get)]:.1f}x")
    print(f"{'='*65}\n")
