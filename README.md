# 🚦 Real-Time Traffic Sign Detection & Recognition

A deep learning-based system for detecting and classifying **43 categories** of traffic signs in real-time using Convolutional Neural Networks (CNN) and OpenCV — trained on the German Traffic Sign Recognition Benchmark (GTSRB).

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-FF6F00?logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-5C3EE8?logo=opencv&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.0+-000000?logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Problem Statement

Traffic sign recognition is a critical component of **autonomous driving** and **Advanced Driver Assistance Systems (ADAS)**. Failure to detect and interpret traffic signs can lead to accidents, violations, and loss of life. This project builds a deep learning-based traffic sign classification system capable of recognizing **43 categories** of German traffic signs from images or live webcam feeds.

## 💡 Motivation

- **Road Safety**: ~1.35 million people die annually in road accidents globally (WHO). Automated sign recognition reduces human error.
- **Autonomous Vehicles**: Essential perception module for self-driving cars (Tesla, Waymo, etc.).
- **Accessibility**: Assists visually impaired or distracted drivers with real-time sign alerts.
- **Smart Cities**: Enables intelligent traffic management and infrastructure monitoring.

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **Custom CNN Model** | 3-block convolutional architecture with BatchNorm, Dropout, and Global Average Pooling |
| 📊 **Complete Training Pipeline** | Data augmentation, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint callbacks |
| 📈 **Comprehensive Evaluation** | Confusion matrix, per-class accuracy, classification report, misclassified sample analysis |
| 🔍 **Single Image Prediction** | Predict any traffic sign image with top-5 confidence scores and visualization |
| 📹 **Real-Time Webcam Detection** | Live traffic sign classification via OpenCV with FPS counter and ROI overlay |
| 🌐 **Web Application** | Modern Flask web UI with drag-and-drop upload, confidence bars, and top-5 results |
| 📦 **Modular Codebase** | Clean, documented, production-ready code with clear separation of concerns |

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python 3.8+** | Core programming language |
| **TensorFlow / Keras** | Deep learning framework (model building & training) |
| **OpenCV** | Image processing & real-time webcam capture |
| **NumPy / Pandas** | Numerical computation & data manipulation |
| **Matplotlib / Seaborn** | Training curves, confusion matrices, data visualization |
| **scikit-learn** | Classification metrics, train-test splitting |
| **Flask** | Web demo interface with REST API |
| **Pillow** | Image loading and preprocessing |

## 📁 Project Structure

```
traffic-sign-recognition/
│
├── src/                          # Core source code
│   ├── __init__.py               # Package initializer
│   ├── config.py                 # Hyperparameters, paths, class labels
│   ├── data_loader.py            # Dataset loading, preprocessing, augmentation
│   ├── model.py                  # CNN architecture definition
│   ├── train.py                  # Training pipeline with callbacks
│   ├── evaluate.py               # Evaluation metrics & visualizations
│   ├── predict.py                # Single image prediction with top-K
│   ├── realtime_detect.py        # Webcam real-time detection (OpenCV)
│   └── utils.py                  # Visualization helpers & dataset stats
│
├── web/                          # Flask web application
│   ├── app.py                    # Flask routes & server
│   ├── templates/
│   │   └── index.html            # Web UI (drag-and-drop upload)
│   └── static/
│       └── style.css             # Modern dark-themed styling
│
├── dataset/                      # Dataset directory
│   └── README.md                 # Download instructions for GTSRB
│
├── models/                       # Saved trained models
│   └── .gitkeep
│
├── outputs/                      # Training plots, metrics, reports
│   └── .gitkeep
│
├── main.py                       # CLI entry point (train/evaluate/predict/webcam/web)
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/traffic-sign-recognition.git
cd traffic-sign-recognition
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Dataset

Download the **GTSRB** dataset from one of these sources:

- **Kaggle**: [GTSRB Dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- **Official**: [INI Benchmark](https://benchmark.ini.rub.de/gtsrb_dataset.html)

Extract into the `dataset/` directory. See [`dataset/README.md`](dataset/README.md) for detailed instructions.

Expected structure after download:
```
dataset/
├── Train/
│   ├── 0/
│   ├── 1/
│   │   ...
│   └── 42/
├── Test/
├── Train.csv
└── Test.csv
```

## 📖 Usage

All commands are run through `main.py`:

### Train the Model
```bash
python main.py train
```
Trains the CNN on the GTSRB dataset with data augmentation. Saves the best model to `models/` and training curves to `outputs/`.

### Evaluate the Model
```bash
python main.py evaluate
```
Generates a full evaluation report: confusion matrix, per-class accuracy, classification report, and misclassified sample visualization.

### Predict a Single Image
```bash
python main.py predict path/to/traffic_sign.png
```
Classifies a single image and displays top-5 predictions with confidence scores.

### Real-Time Webcam Detection
```bash
python main.py webcam
```
Opens the webcam feed and classifies traffic signs in real-time. Position a traffic sign in the green ROI box. Press **Q** to quit.

### Web Application
```bash
python main.py web
```
Launches a Flask web app at `http://localhost:5000` with a drag-and-drop interface for uploading and classifying traffic sign images.

### Model Summary
```bash
python main.py summary
```
Prints the CNN architecture and parameter count.

## 🧠 Model Architecture

```
TrafficSignNet — Custom CNN
════════════════════════════════════════════
 Input: 32 × 32 × 3 (RGB)
────────────────────────────────────────────
 Conv Block 1: 2×Conv2D(32) → BN → ReLU → MaxPool → Dropout(0.25)
 Conv Block 2: 2×Conv2D(64) → BN → ReLU → MaxPool → Dropout(0.25)
 Conv Block 3: 2×Conv2D(128) → BN → ReLU → MaxPool → Dropout(0.25)
────────────────────────────────────────────
 Global Average Pooling
 Dense(256) → BN → ReLU → Dropout(0.5)
 Dense(128) → BN → ReLU → Dropout(0.3)
 Dense(43)  → Softmax
════════════════════════════════════════════
```

**Key Design Choices:**
- **BatchNormalization** after every layer stabilizes training
- **Dropout** (0.25–0.5) prevents overfitting on underrepresented classes
- **Global Average Pooling** reduces parameters compared to Flatten
- **Data Augmentation**: rotation, zoom, shift, shear, brightness adjustments

## 📊 Dataset Details

**German Traffic Sign Recognition Benchmark (GTSRB)**

| Property | Value |
|---|---|
| Training images | ~39,209 |
| Test images | ~12,630 |
| Number of classes | 43 |
| Image format | PPM / PNG |
| Image sizes | 15×15 to 250×250 (resized to 32×32) |

The 43 classes include speed limit signs, prohibition signs, warning signs, mandatory signs, and information signs.

## 🔮 Future Improvements

- [ ] **Transfer Learning**: Use pretrained models (ResNet50, MobileNetV2) for improved accuracy
- [ ] **Object Detection**: Integrate YOLO/SSD for traffic sign localization in full scenes
- [ ] **Mobile Deployment**: Convert to TFLite for Android/iOS deployment
- [ ] **Multi-Scale Detection**: Handle signs at various distances and scales
- [ ] **Weather Robustness**: Train with synthetic rain, fog, and night augmentations
- [ ] **Video Pipeline**: Process dashcam video files with tracking across frames
- [ ] **Edge Deployment**: Optimize for Raspberry Pi / Jetson Nano
- [ ] **Dataset Expansion**: Include signs from other countries (US, India, EU)

## 📜 References

1. Stallkamp, J., Schlipsing, M., Salmen, J., & Igel, C. (2012). *Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition*. Neural Networks, 32, 323–332.
2. GTSRB Benchmark: https://benchmark.ini.rub.de/gtsrb_dataset.html
3. TensorFlow Documentation: https://www.tensorflow.org/
4. OpenCV Documentation: https://docs.opencv.org/

## 📝 License

This project is developed as part of a **Computer Vision course (BYOP)**.

---

**Built with** ❤️ **using TensorFlow, OpenCV & Flask**
