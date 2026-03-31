# ============================================================================
# predict.py — Single Image Prediction with Confidence Scores
# ============================================================================
"""
Load a trained model and predict the traffic sign class from a single image.
Supports:
  • Single image prediction via CLI
  • Top-K predictions with confidence scores
  • Prediction visualization with annotated image
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from src.config import (
    MODEL_SAVE_PATH, IMG_HEIGHT, IMG_WIDTH,
    NUM_CLASSES, get_class_name, OUTPUT_DIR,
    ensure_directories,
)
from src.data_loader import load_image, preprocess_image


def predict_single_image(image_path: str, model=None, top_k: int = 5) -> dict:
    """
    Predict the traffic sign class from a single image file.

    Args:
        image_path: Path to the input image.
        model:      Pre-loaded Keras model (loaded from disk if None).
        top_k:      Number of top predictions to return.

    Returns:
        Dictionary containing:
          - 'predicted_class': int
          - 'class_name': str
          - 'confidence': float
          - 'top_k': list of (class_id, class_name, probability)
    """
    # Load model if not provided
    if model is None:
        if not os.path.exists(MODEL_SAVE_PATH):
            raise FileNotFoundError(
                f"No trained model at '{MODEL_SAVE_PATH}'. "
                "Run training first: python main.py train"
            )
        model = load_model(MODEL_SAVE_PATH)

    # Load and preprocess image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    raw_image = load_image(image_path)
    processed = preprocess_image(raw_image)
    input_tensor = np.expand_dims(processed, axis=0)  # Add batch dimension

    # Predict
    predictions = model.predict(input_tensor, verbose=0)[0]

    # Get top-K predictions
    top_k_indices = np.argsort(predictions)[::-1][:top_k]
    top_k_results = [
        (int(idx), get_class_name(int(idx)), float(predictions[idx]))
        for idx in top_k_indices
    ]

    predicted_class = int(np.argmax(predictions))
    result = {
        'predicted_class': predicted_class,
        'class_name': get_class_name(predicted_class),
        'confidence': float(predictions[predicted_class]),
        'top_k': top_k_results,
        'raw_image': raw_image,
    }

    return result


def visualize_prediction(result: dict, image_path: str, save: bool = True) -> None:
    """
    Create a visual display of the prediction:
    Left panel  — the input image with predicted label
    Right panel — horizontal bar chart of top-5 confidence scores

    Args:
        result:     Prediction result dictionary from predict_single_image().
        image_path: Original image path (for the title).
        save:       Whether to save the visualization.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                              gridspec_kw={'width_ratios': [1, 1.5]})

    # ── Left: Input Image ────────────────────────────────────────────
    axes[0].imshow(result['raw_image'])
    axes[0].set_title(
        f"Prediction: {result['class_name']}\n"
        f"Confidence: {result['confidence']*100:.1f}%",
        fontsize=11, fontweight='bold',
        color='green' if result['confidence'] > 0.8 else 'orange',
    )
    axes[0].axis('off')

    # ── Right: Top-K Confidence Bar Chart ────────────────────────────
    labels = [f"[{c[0]}] {c[1]}" for c in result['top_k']]
    probs  = [c[2] * 100 for c in result['top_k']]
    colors = ['#4CAF50' if i == 0 else '#90CAF9' for i in range(len(probs))]

    y_pos = range(len(labels))
    axes[1].barh(y_pos, probs, color=colors, edgecolor='white', height=0.6)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(labels, fontsize=9)
    axes[1].set_xlabel('Confidence (%)')
    axes[1].set_title('Top-5 Predictions', fontsize=11, fontweight='bold')
    axes[1].set_xlim([0, 105])
    axes[1].invert_yaxis()

    # Add percentage labels on bars
    for i, v in enumerate(probs):
        axes[1].text(v + 1, i, f'{v:.1f}%', va='center', fontsize=9)

    plt.tight_layout()

    if save:
        ensure_directories()
        save_path = os.path.join(OUTPUT_DIR, "prediction_result.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Prediction visualization saved to: {save_path}")

    plt.close()


def run_prediction(image_path: str) -> None:
    """
    CLI-friendly prediction function: predict and display results.

    Args:
        image_path: Path to the image file to classify.
    """
    print("=" * 60)
    print("  🔍 TRAFFIC SIGN PREDICTION")
    print("=" * 60)
    print(f"  Image: {image_path}")
    print("-" * 60)

    result = predict_single_image(image_path)

    print(f"\n  ✅ Predicted Class : {result['class_name']}")
    print(f"  📊 Confidence     : {result['confidence']*100:.2f}%")
    print(f"\n  Top-5 Predictions:")
    print(f"  {'Rank':<6}{'Class':<45}{'Confidence':<12}")
    print(f"  {'-'*60}")
    for rank, (cls_id, cls_name, prob) in enumerate(result['top_k'], 1):
        marker = "  ◀" if rank == 1 else ""
        print(f"  {rank:<6}{cls_name:<45}{prob*100:>6.2f}%{marker}")

    visualize_prediction(result, image_path)
    print(f"\n{'='*60}")


# ──────────────────────────────────────────────────────────────────────
# Run prediction if executed directly
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.predict <image_path>")
        sys.exit(1)
    run_prediction(sys.argv[1])
