#!/usr/bin/env python3
# ============================================================================
# main.py — Central Entry Point for SignSightAI Project
# ============================================================================
"""
Command-line interface to run all major project operations:

Usage:
    python main.py train       Train the CNN model on the GTSRB dataset
    python main.py evaluate    Evaluate the trained model on the test set
    python main.py predict     Predict the class of a single image
    python main.py webcam      Launch real-time webcam detection
    python main.py web         Start the Flask web application
    python main.py summary     Display CNN model architecture summary

Examples:
    python main.py train
    python main.py evaluate
    python main.py predict path/to/image.png
    python main.py webcam
    python main.py web
"""

import sys
import os

# Ensure the project root is in the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_banner():
    """Display a styled project banner in the terminal."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   🚦  SIGNSIGHTAI: REAL-TIME TRAFFIC SIGN DETECTION    🚦    ║
    ║                                                              ║
    ║   CNN-based classification of 43 traffic sign categories     ║
    ║   using the German Traffic Sign Recognition Benchmark        ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_usage():
    """Display available commands and usage instructions."""
    print("""
    Available Commands:
    ─────────────────────────────────────────────────────────────
      train      Train the CNN model on the GTSRB dataset
      evaluate   Evaluate the trained model with metrics & plots
      predict    Classify a single traffic sign image
      webcam     Real-time detection from webcam feed
      web        Launch the Flask web application
      summary    Print the CNN model architecture summary
    ─────────────────────────────────────────────────────────────

    Usage:
      python main.py <command> [arguments]

    Examples:
      python main.py train
      python main.py evaluate
      python main.py predict dataset/Test/00000.png
      python main.py webcam
      python main.py web
    """)


def cmd_train():
    """Execute the training pipeline."""
    from src.train import train_model
    train_model()


def cmd_evaluate():
    """Execute the evaluation pipeline."""
    from src.evaluate import evaluate_model
    evaluate_model()


def cmd_predict(image_path: str):
    """Predict the class of a single image."""
    from src.predict import run_prediction
    run_prediction(image_path)


def cmd_webcam():
    """Launch the real-time webcam detection."""
    from src.realtime_detect import run_webcam_detection
    run_webcam_detection()


def cmd_web():
    """Start the Flask web application."""
    from web.app import run_web_app
    run_web_app()


def cmd_summary():
    """Print the model architecture summary."""
    from src.model import print_model_summary
    print_model_summary()


# ──────────────────────────────────────────────────────────────────────
# Main CLI Dispatcher
# ──────────────────────────────────────────────────────────────────────

def main():
    """Parse command-line arguments and dispatch to the appropriate handler."""
    print_banner()

    if len(sys.argv) < 2:
        print("  [ERROR] No command specified.\n")
        print_usage()
        sys.exit(1)

    command = sys.argv[1].lower().strip()

    if command == "train":
        cmd_train()

    elif command == "evaluate":
        cmd_evaluate()

    elif command == "predict":
        if len(sys.argv) < 3:
            print("  [ERROR] Please provide an image path.")
            print("  Usage:  python main.py predict <image_path>")
            sys.exit(1)
        image_path = sys.argv[2]
        cmd_predict(image_path)

    elif command == "webcam":
        cmd_webcam()

    elif command == "web":
        cmd_web()

    elif command == "summary":
        cmd_summary()

    elif command in ("--help", "-h", "help"):
        print_usage()

    else:
        print(f"  [ERROR] Unknown command: '{command}'\n")
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
