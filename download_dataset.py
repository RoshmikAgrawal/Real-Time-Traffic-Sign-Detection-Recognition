"""
Download the GTSRB dataset from a public source.
This script downloads and extracts the dataset into the dataset/ directory.
"""

import os
import sys
import zipfile
import urllib.request
import shutil

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

# GTSRB dataset URLs (public mirrors)
TRAIN_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"
TEST_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip"
TEST_GT_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip"


def download_file(url, dest_path):
    """Download a file with progress display."""
    print(f"[INFO] Downloading: {os.path.basename(dest_path)}")
    print(f"       URL: {url}")
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb_done = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r       Progress: {pct:.1f}% ({mb_done:.1f}/{mb_total:.1f} MB)")
            sys.stdout.flush()
    
    urllib.request.urlretrieve(url, dest_path, reporthook=progress_hook)
    print()  # newline after progress


def extract_zip(zip_path, extract_to):
    """Extract a zip file."""
    print(f"[INFO] Extracting: {os.path.basename(zip_path)}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)
    print(f"       Done.")


def reorganize_train_data():
    """
    Reorganize the extracted GTSRB training data into the expected structure:
    dataset/Train/0/, dataset/Train/1/, ..., dataset/Train/42/
    """
    # The extracted data is in GTSRB/Final_Training/Images/XXXXX/
    source_dir = os.path.join(DATASET_DIR, "GTSRB", "Final_Training", "Images")
    target_dir = os.path.join(DATASET_DIR, "Train")
    
    if not os.path.exists(source_dir):
        print(f"[WARN] Source directory not found: {source_dir}")
        # Try alternate path
        source_dir = os.path.join(DATASET_DIR, "GTSRB", "Training", "Images")
        if not os.path.exists(source_dir):
            print(f"[WARN] Alternate source also not found. Checking existing structure...")
            if os.path.exists(target_dir):
                print("[INFO] Train directory already exists.")
            return
    
    print(f"[INFO] Reorganizing training data...")
    os.makedirs(target_dir, exist_ok=True)
    
    for class_dir in sorted(os.listdir(source_dir)):
        src = os.path.join(source_dir, class_dir)
        if os.path.isdir(src):
            # Convert folder name like "00005" to "5"
            try:
                class_id = str(int(class_dir))
            except ValueError:
                class_id = class_dir
            
            dst = os.path.join(target_dir, class_id)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            
            # Count images
            img_count = len([f for f in os.listdir(dst) 
                           if f.lower().endswith(('.png', '.jpg', '.ppm'))])
            print(f"       Class {class_id:>2}: {img_count} images")
    
    print(f"[INFO] Training data reorganized into: {target_dir}")


def reorganize_test_data():
    """
    Reorganize extracted test data into dataset/Test/ with a Test.csv.
    """
    source_dir = os.path.join(DATASET_DIR, "GTSRB", "Final_Test", "Images")
    target_dir = os.path.join(DATASET_DIR, "Test")
    
    if not os.path.exists(source_dir):
        print(f"[WARN] Test source not found: {source_dir}")
        if os.path.exists(target_dir):
            print("[INFO] Test directory already exists.")
        return
    
    print(f"[INFO] Reorganizing test data...")
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)
    
    img_count = len([f for f in os.listdir(target_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.ppm'))])
    print(f"[INFO] Test data: {img_count} images in {target_dir}")


def create_test_csv():
    """
    Create Test.csv from the ground truth file if available.
    Maps image paths to class IDs.
    """
    gt_file = os.path.join(DATASET_DIR, "GT-final_test.csv")
    if not os.path.exists(gt_file):
        # Try alternate name
        gt_file = os.path.join(DATASET_DIR, "GTSRB", "GT-final_test.csv")
    
    if not os.path.exists(gt_file):
        print("[WARN] Ground truth CSV not found. Test.csv not created.")
        return
    
    import csv
    
    print("[INFO] Creating Test.csv...")
    rows = []
    with open(gt_file, 'r') as f:
        # GTSRB GT file uses semicolons
        reader = csv.reader(f, delimiter=';')
        header = next(reader)
        for row in reader:
            filename = row[0]
            class_id = row[-1]
            rows.append({
                'Width': row[1] if len(row) > 1 else '',
                'Height': row[2] if len(row) > 2 else '',
                'Roi.X1': row[3] if len(row) > 3 else '',
                'Roi.Y1': row[4] if len(row) > 4 else '',
                'Roi.X2': row[5] if len(row) > 5 else '',
                'Roi.Y2': row[6] if len(row) > 6 else '',
                'ClassId': int(class_id),
                'Path': f"Test/{filename}",
            })
    
    test_csv_path = os.path.join(DATASET_DIR, "Test.csv")
    with open(test_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Width', 'Height', 'Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2', 'ClassId', 'Path'])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"[INFO] Test.csv created with {len(rows)} entries.")


def main():
    print("=" * 60)
    print("  GTSRB Dataset Downloader")
    print("=" * 60)
    
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    # Check if data already exists
    train_dir = os.path.join(DATASET_DIR, "Train")
    if os.path.exists(train_dir) and len(os.listdir(train_dir)) >= 43:
        print("[INFO] Training data already exists. Skipping download.")
        print(f"       Found {len(os.listdir(train_dir))} class directories.")
        return
    
    # Download training data
    train_zip = os.path.join(DATASET_DIR, "train.zip")
    if not os.path.exists(train_zip):
        download_file(TRAIN_URL, train_zip)
    extract_zip(train_zip, DATASET_DIR)
    reorganize_train_data()
    
    # Download test data
    test_zip = os.path.join(DATASET_DIR, "test.zip")
    if not os.path.exists(test_zip):
        download_file(TEST_URL, test_zip)
    extract_zip(test_zip, DATASET_DIR)
    reorganize_test_data()
    
    # Download test ground truth
    gt_zip = os.path.join(DATASET_DIR, "test_gt.zip")
    if not os.path.exists(gt_zip):
        download_file(TEST_GT_URL, gt_zip)
    extract_zip(gt_zip, DATASET_DIR)
    create_test_csv()
    
    # Cleanup zip files
    print("\n[INFO] Cleaning up zip files...")
    for zf in [train_zip, test_zip, gt_zip]:
        if os.path.exists(zf):
            os.remove(zf)
    
    # Cleanup extracted GTSRB directory  
    gtsrb_dir = os.path.join(DATASET_DIR, "GTSRB")
    if os.path.exists(gtsrb_dir):
        shutil.rmtree(gtsrb_dir)
    
    print("\n" + "=" * 60)
    print("  ✅ Dataset download complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
