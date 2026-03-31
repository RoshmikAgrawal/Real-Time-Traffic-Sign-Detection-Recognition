# ============================================================================
# app.py — Flask Web Application for SignSightAI
# ============================================================================
"""
A sleek web interface for uploading traffic sign images and viewing
predictions from the trained CNN model.

Features:
  • Drag-and-drop image upload
  • Top-5 predictions with confidence bars
  • Responsive modern UI
  • REST API endpoint for programmatic access

Routes:
  GET  /           → Web interface
  POST /predict    → Upload image and get prediction
  GET  /api/health → Health check endpoint
"""

import os
import sys
import json
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import MODEL_SAVE_PATH, IMG_HEIGHT, IMG_WIDTH, get_class_name
from src.data_loader import load_image, preprocess_image

# ──────────────────────────────────────────────────────────────────────
# Flask App Configuration
# ──────────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload
app.config['UPLOAD_FOLDER'] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'uploads'
)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'ppm', 'bmp', 'webp'}

# Global model variable (loaded once at startup)
model = None


def allowed_file(filename: str) -> bool:
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model_once():
    """Load the trained model into memory (singleton pattern)."""
    global model
    if model is None:
        if not os.path.exists(MODEL_SAVE_PATH):
            raise FileNotFoundError(
                f"Trained model not found at '{MODEL_SAVE_PATH}'. "
                "Train the model first: python main.py train"
            )
        print("[INFO] Loading trained model...")
        model = load_model(MODEL_SAVE_PATH)
        print("[INFO] Model loaded successfully!")
    return model


# ──────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the main web interface."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle image upload and return prediction results.

    Accepts: multipart/form-data with 'file' field containing an image.
    Returns: JSON with prediction results or error message.
    """
    try:
        loaded_model = load_model_once()
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 500

    # Validate upload
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({
            'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400

    # Save uploaded file temporarily
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # Load, preprocess, and predict
        raw_image = load_image(filepath)
        processed = preprocess_image(raw_image)
        input_tensor = np.expand_dims(processed, axis=0)

        predictions = loaded_model.predict(input_tensor, verbose=0)[0]

        # Get top-5 predictions
        top5_indices = np.argsort(predictions)[::-1][:5]
        results = []
        for idx in top5_indices:
            results.append({
                'class_id': int(idx),
                'class_name': get_class_name(int(idx)),
                'confidence': float(predictions[idx]) * 100,
            })

        predicted_class = int(np.argmax(predictions))

        return jsonify({
            'success': True,
            'predicted_class': predicted_class,
            'class_name': get_class_name(predicted_class),
            'confidence': float(predictions[predicted_class]) * 100,
            'top5': results,
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    finally:
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route('/api/health')
def health_check():
    """Health check endpoint for monitoring."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
    })


# ──────────────────────────────────────────────────────────────────────
# Run the Flask App
# ──────────────────────────────────────────────────────────────────────

def run_web_app(host: str = '0.0.0.0', port: int = 5000, debug: bool = True):
    """Start the Flask web server."""
    print("\n" + "=" * 60)
    print("  🌐 SIGNSIGHTAI — WEB INTERFACE")
    print("=" * 60)
    print(f"  URL: http://localhost:{port}")
    print(f"  Press Ctrl+C to stop the server.")
    print("=" * 60 + "\n")
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_web_app()
