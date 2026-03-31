# SignSightAI — Traffic Sign Detection System in Real Time

SignSightAI is a real-time computer vision system that detects and classifies German traffic signs (43 categories) using a web camera or uploaded images. It is equipped with a custom Convolutional Neural Network (CNN) that processes image shapes, colors, and contours independently, leading to strong, reliable real-time traffic sign classification.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-FF6F00?logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-5C3EE8?logo=opencv&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.0+-000000?logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Why does this matter?

Traffic sign recognition is a critical component of autonomous driving and Advanced Driver Assistance Systems (ADAS). In the real world, failure to detect and interpret traffic signs can lead to severe accidents, traffic violations, and loss of life. An inexpensive, camera-based automated sign recognition system reduces human error and paves the way for intelligent traffic management systems.

## Features

| Feature | Description |
|---|---|
| **Custom CNN Classification** | Uses a 3-block convolutional architecture optimized for lightweight, real-time inference across 43 classes. |
| **Real-Time Web Detection** | Tracks and classifies live video feed using OpenCV, automatically filtering results by confidence. |
| **Smart Web Dashboard** | Modern, dark-themed Flask web UI with drag-and-drop upload and visual prediction confidence bars. |
| **Complete Training Pipeline** | Includes integrated callbacks like EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint alongside data augmentation. |
| **Comprehensive Evaluation** | Generates confusion matrices, visualizes misclassified samples, and calculates per-class precision and recall. |
| **Adjustable Configurations** | Tune image sizes, dataset properties, data augmentation logic, and training hyperparameters directly via `config.py`. |

## How It Works

### The Convolutional Neural Network (CNN)

The network is designed specifically to capture visual elements indicative of traffic signs:

*   **Low to Mid-level feature extraction:** The first two Conv2D blocks capture primal shapes (circles, triangles, octagons) and colors along with edge borders.
*   **High-level feature extraction:** The third Conv2D block specifically focuses on sign-specific symbols and interior patterns (digits, pedestrian icons).

The network finalizes classifications using **Global Average Pooling** (reducing parameter bloat) followed by fully connected layers and a Softmax output layer representing probabilities for each of the 43 sign categories.

### Data Augmentation
To simulate real-world environmental disruptions, the model receives altered images automatically during training using:
*   Rotation (±15°)
*   Height & Width shifts (10%)
*   Shear & Zoom (up to 20%)
*   Brightness corrections (0.8x to 1.2x)

### Real-time Classification
In a live webcam feed, the Region of Interest (ROI) is cropped, resized to `32x32`, normalized strictly to `0-1`, and run through the loaded `traffic_sign_model.keras` in milliseconds. The highest probability class above a strict confidence limit triggers the on-screen alert.

## Project Structure

```text
SignSightAI/
├── src/
│   ├── __init__.py        # Package marker
│   ├── config.py          # All tunable parameters & constants
│   ├── data_loader.py     # Dataset loading, preprocessing, augmentation
│   ├── model.py           # CNN architecture definition
│   ├── train.py           # Training pipeline
│   ├── predict.py         # Single image inference
│   ├── realtime_detect.py # OpenCV webcam integration
│   ├── evaluate.py        # Confusion matrices & evaluation
│   └── utils.py           # Utility scripts
├── web/
│   ├── app.py             # Flask Web Application logic
│   ├── static/
│   │   └── style.css      # Dark-themed GUI stylesheet
│   └── templates/
│       └── index.html     # Client-side web interface
├── dataset/               # Auto-populates GTSRB structured dataset
├── models/                # Saved trained Keras models
├── outputs/               # Saved charts, graphs, and logs
├── main.py                # Command Line Entry Point
├── README.md              
└── requirements.txt       # Python dependencies
```

## Getting Started

### Prerequisites
*   Python 3.8+
*   A working webcam (for live detection)
*   Windows / macOS / Linux

### Installation

Clone the repository
```bash
git clone https://github.com/RoshmikAgrawal/Real-Time-Traffic-Sign-Detection-Recognition.git
cd Real-Time-Traffic-Sign-Detection-Recognition
```

Create virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

Install dependencies
```bash
pip install -r requirements.txt
```

*(Note: Data must be gathered prior to model training. See dataset execution scripts or run model evaluations as specified below).*

## Running the Application

There are multiple ways to operate this project. Start the Flask application GUI via:

```bash
python main.py web
```
The application will launch. Open a browser and navigate to `http://localhost:5000` to interact.

Alternatively, to jump straight into active live-feed surveillance:
```bash
python main.py webcam
```

## Running Evaluations

To run the robust numerical testing and output comprehensive statistics of the model:
```bash
python main.py evaluate
```
This triggers the visual mapping sequence, calculating class loss metrics and pushing visual `.png` matrix reports to `outputs/`.

## Configuration

You can adjust all thresholds directly in the `src/config.py` parameters list:

| Parameter | Default | Description |
|---|---|---|
| `IMG_HEIGHT` \ `IMG_WIDTH` | 32 | Native dimensions the CNN accepts for prediction. |
| `BATCH_SIZE` | 64 | Mini-batch sizing limit utilized for network updates per epoch. |
| `EPOCHS` | 30 | Absolute cap count on training iterations over the dataset. |
| `LEARNING_RATE` | 0.001 | Base iteration descent value assigned to the Adam optimizer. |
| `VALIDATION_SPLIT` | 0.2 | Total fraction of data isolated exclusively for unbiased test loops. |
| `NUM_CLASSES` | 43 | Total traffic categories the dataset supports and classifies. |

*(Also supports explicit threshold shifts, rotation limiters, and hardware indices natively through related web modules).*

## Tech Stack

| Component | Technology |
|---|---|
| **Language** | Python 3.8+ |
| **Deep Learning**| TensorFlow / Keras |
| **Video Processing** | OpenCV |
| **GUI & API** | Flask |
| **Math & Data** | NumPy, Pandas |
| **Metrics & Viz** | SciKit-Learn, Matplotlib, Seaborn |

## License

This project is licensed under the MIT License.

## Author

Roshmik Agrawal

## Future Improvements
*   Implement Transfer Learning (e.g., using a backbone like MobileNetV2) to achieve higher benchmark metrics
*   Introduce Object Detection (YOLO/SSD) to locate the traffic sign within moving automotive sequences automatically
*   Mobile deployment framework conversion through TFLite / ONNX formats
*   Simulate varied weather (rain, fog, low light) locally for enhanced environmental robustness
