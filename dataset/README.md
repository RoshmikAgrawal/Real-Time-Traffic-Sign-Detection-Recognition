# Dataset: German Traffic Sign Recognition Benchmark (GTSRB)

## Overview

The GTSRB dataset contains **50,000+ images** of **43 classes** of German traffic signs.
It is the standard benchmark for traffic sign classification research.

## Download Instructions

### Option 1: Kaggle (Recommended)

1. Visit: https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
2. Download the dataset
3. Extract into this `dataset/` directory

### Option 2: Official Source

1. Visit: https://benchmark.ini.rub.de/gtsrb_dataset.html
2. Download the training and test images
3. Extract into this `dataset/` directory

### Option 3: Command Line (with Kaggle API)

```bash
# Install and configure Kaggle API
pip install kaggle

# Download the dataset
kaggle datasets download -d meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
unzip gtsrb-german-traffic-sign.zip -d dataset/
```

## Expected Directory Structure

After downloading and extracting, the directory structure should look like:

```
dataset/
├── Train/
│   ├── 0/          # Speed limit (20km/h) images
│   ├── 1/          # Speed limit (30km/h) images
│   ├── 2/          # Speed limit (50km/h) images
│   │   ...
│   └── 42/         # End of no passing by vehicles over 3.5t
├── Test/
│   ├── 00000.png
│   ├── 00001.png
│   │   ...
│   └── XXXXX.png
├── Train.csv       # Training labels and metadata
├── Test.csv        # Test labels and metadata
├── Meta.csv        # Class metadata
└── README.md       # This file
```

## Dataset Statistics

| Property          | Value            |
|-------------------|------------------|
| Training images   | ~39,209          |
| Test images       | ~12,630          |
| Number of classes | 43               |
| Image format      | PPM / PNG        |
| Image size        | Varies (15×15 to 250×250) |
| Color space       | RGB              |

## Classes

The dataset includes 43 types of traffic signs, such as:
- Speed limit signs (20, 30, 50, 60, 70, 80, 100, 120 km/h)
- Prohibition signs (No passing, No entry, No vehicles)
- Warning signs (Dangerous curves, Bumpy road, Slippery road)
- Mandatory signs (Turn left/right, Go straight, Roundabout)
- Information signs (Priority road, Yield, Stop)

## Citation

```bibtex
@inproceedings{stallkamp2012gtsrb,
  title={Man vs. computer: Benchmarking machine learning algorithms
         for traffic sign recognition},
  author={Stallkamp, J. and Schlipsing, M. and Salmen, J. and Igel, C.},
  journal={Neural Networks},
  volume={32},
  pages={323--332},
  year={2012}
}
```
