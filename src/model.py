# ============================================================================
# model.py — CNN Architecture for Traffic Sign Classification
# ============================================================================
"""
Defines a custom Convolutional Neural Network (CNN) architecture
optimized for traffic sign classification on 32×32 RGB images.

Architecture Overview:
  ┌─────────────────────────────────────────────┐
  │  Input: 32 × 32 × 3 (RGB image)            │
  ├─────────────────────────────────────────────┤
  │  Conv Block 1: 32 filters → BN → ReLU → MP │
  │  Conv Block 2: 64 filters → BN → ReLU → MP │
  │  Conv Block 3: 128 filters → BN → ReLU → MP│
  ├─────────────────────────────────────────────┤
  │  Global Average Pooling                     │
  │  Dense 256 → BN → ReLU → Dropout(0.5)      │
  │  Dense 128 → BN → ReLU → Dropout(0.3)      │
  │  Dense 43  → Softmax                        │
  └─────────────────────────────────────────────┘

Key design decisions:
  • BatchNormalization after every conv/dense layer stabilizes training
  • Dropout (0.25–0.5) prevents overfitting on smaller classes
  • Global Average Pooling reduces parameters vs. Flatten
  • ~500K parameters — lightweight enough for real-time inference
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling2D,
    Input,
)
from tensorflow.keras.optimizers import Adam

from src.config import INPUT_SHAPE, NUM_CLASSES, LEARNING_RATE


def build_traffic_sign_model(input_shape: tuple = INPUT_SHAPE,
                              num_classes: int = NUM_CLASSES,
                              learning_rate: float = LEARNING_RATE):
    """
    Build and compile the CNN model for traffic sign classification.

    Args:
        input_shape:   Tuple (height, width, channels) for input images.
        num_classes:   Number of output classes (43 for GTSRB).
        learning_rate: Initial learning rate for Adam optimizer.

    Returns:
        A compiled Keras Sequential model.
    """
    model = Sequential(name="TrafficSignNet")

    # ── Input Layer ──────────────────────────────────────────────────
    model.add(Input(shape=input_shape))

    # ── Convolutional Block 1 ────────────────────────────────────────
    # Extract low-level features: edges, corners, color gradients
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu',
                     name='conv1a'))
    model.add(BatchNormalization(name='bn1a'))
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu',
                     name='conv1b'))
    model.add(BatchNormalization(name='bn1b'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool1'))
    model.add(Dropout(0.25, name='drop1'))

    # ── Convolutional Block 2 ────────────────────────────────────────
    # Extract mid-level features: shapes, patterns, sign contours
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu',
                     name='conv2a'))
    model.add(BatchNormalization(name='bn2a'))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu',
                     name='conv2b'))
    model.add(BatchNormalization(name='bn2b'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool2'))
    model.add(Dropout(0.25, name='drop2'))

    # ── Convolutional Block 3 ────────────────────────────────────────
    # Extract high-level features: sign-specific symbols, text, icons
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu',
                     name='conv3a'))
    model.add(BatchNormalization(name='bn3a'))
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu',
                     name='conv3b'))
    model.add(BatchNormalization(name='bn3b'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool3'))
    model.add(Dropout(0.25, name='drop3'))

    # ── Classification Head ──────────────────────────────────────────
    # Global Average Pooling reduces spatial dims to 1×1 per filter
    model.add(GlobalAveragePooling2D(name='gap'))

    # Fully connected layers for final classification
    model.add(Dense(256, activation='relu', name='fc1'))
    model.add(BatchNormalization(name='bn_fc1'))
    model.add(Dropout(0.5, name='drop_fc1'))

    model.add(Dense(128, activation='relu', name='fc2'))
    model.add(BatchNormalization(name='bn_fc2'))
    model.add(Dropout(0.3, name='drop_fc2'))

    # Output layer — softmax for multi-class probability distribution
    model.add(Dense(num_classes, activation='softmax', name='output'))

    # ── Compile ──────────────────────────────────────────────────────
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model


def print_model_summary():
    """Build the model and print its architecture summary."""
    model = build_traffic_sign_model()
    model.summary()
    total_params = model.count_params()
    print(f"\n📊 Total parameters: {total_params:,}")
    print(f"📊 Estimated model size: {total_params * 4 / (1024**2):.2f} MB (float32)")
    return model


# ──────────────────────────────────────────────────────────────────────
# Quick Test
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print_model_summary()
