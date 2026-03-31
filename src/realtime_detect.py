# ============================================================================
# realtime_detect.py — Real-Time Traffic Sign Detection via Webcam
# ============================================================================
"""
Uses OpenCV to capture video from the webcam and classify traffic signs
in real-time using the trained CNN model.

Features:
  • Live video feed with prediction overlay
  • Confidence threshold filtering (ignore low-confidence detections)
  • FPS counter for performance monitoring
  • Region of Interest (ROI) box to guide sign placement
  • Press 'q' to quit the live feed

Note: This module requires a webcam connected to the system.
"""

import cv2
import time
import numpy as np
from tensorflow.keras.models import load_model

from src.config import (
    MODEL_SAVE_PATH, IMG_HEIGHT, IMG_WIDTH,
    get_class_name,
)
from src.data_loader import preprocess_image


# ──────────────────────────────────────────────────────────────────────
# Configuration for Real-Time Detection
# ──────────────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.60   # Minimum confidence to display prediction
ROI_SIZE             = 200    # Size of the Region of Interest box (pixels)
FONT                 = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE           = 0.7
FONT_THICKNESS       = 2


def run_webcam_detection() -> None:
    """
    Launch real-time traffic sign detection from the webcam.

    Workflow:
      1. Load the trained CNN model
      2. Open webcam capture (device 0)
      3. For each frame:
         a. Extract the ROI (center of frame)
         b. Resize and preprocess the ROI
         c. Predict the traffic sign class
         d. Overlay prediction text and confidence
      4. Press 'q' to exit
    """
    # ── Load Model ───────────────────────────────────────────────────
    print("[INFO] Loading trained model...")
    try:
        model = load_model(MODEL_SAVE_PATH)
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        print("        Train the model first: python main.py train")
        return

    # ── Open Webcam ──────────────────────────────────────────────────
    print("[INFO] Starting webcam capture...")
    print("[INFO] Position a traffic sign in the green box.")
    print("[INFO] Press 'q' to quit.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Check your camera connection.")
        return

    # Get frame dimensions for ROI placement
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate ROI coordinates (centered in frame)
    roi_x1 = (frame_width - ROI_SIZE) // 2
    roi_y1 = (frame_height - ROI_SIZE) // 2
    roi_x2 = roi_x1 + ROI_SIZE
    roi_y2 = roi_y1 + ROI_SIZE

    prev_time = time.time()
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame. Exiting...")
            break

        # ── Calculate FPS ────────────────────────────────────────────
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time + 1e-8)
        prev_time = current_time

        # ── Extract ROI ──────────────────────────────────────────────
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        # Preprocess ROI for model input
        roi_resized = cv2.resize(roi, (IMG_WIDTH, IMG_HEIGHT))
        roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
        roi_normalized = preprocess_image(roi_rgb)
        input_tensor = np.expand_dims(roi_normalized, axis=0)

        # ── Predict ──────────────────────────────────────────────────
        predictions = model.predict(input_tensor, verbose=0)[0]
        class_id = int(np.argmax(predictions))
        confidence = float(predictions[class_id])
        class_name = get_class_name(class_id)

        # ── Draw ROI Box ─────────────────────────────────────────────
        box_color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), box_color, 3)

        # ── Overlay Prediction (only if confidence exceeds threshold) ─
        if confidence >= CONFIDENCE_THRESHOLD:
            # Background rectangle for text readability
            label = f"{class_name}"
            conf_text = f"Confidence: {confidence*100:.1f}%"

            # Prediction label
            cv2.putText(frame, label, (roi_x1, roi_y1 - 30),
                        FONT, FONT_SCALE, (0, 255, 0), FONT_THICKNESS)
            cv2.putText(frame, conf_text, (roi_x1, roi_y1 - 8),
                        FONT, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(frame, "Place a traffic sign in the box",
                        (roi_x1 - 50, roi_y1 - 10),
                        FONT, 0.5, (0, 165, 255), 1)

        # ── FPS Counter ──────────────────────────────────────────────
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    FONT, 0.7, (255, 255, 255), 2)

        # ── Display Frame ────────────────────────────────────────────
        cv2.imshow("Traffic Sign Detection — Press 'q' to Quit", frame)

        # ── Key Press Handling ───────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC to quit
            break

    # ── Cleanup ──────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam detection stopped.")


# ──────────────────────────────────────────────────────────────────────
# Run if executed directly
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_webcam_detection()
