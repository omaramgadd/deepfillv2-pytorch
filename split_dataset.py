import os
import shutil
import random
from pathlib import Path

def split_dataset(source_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split the MIT Indoor dataset into train, validation, and test sets
    
    Args:
        source_dir: Path to the source directory containing images
        output_dir: Path to create train, val, test directories
        train_ratio: Proportion of images for training
        val_ratio: Proportion of images for validation
        test_ratio: Proportion of images for testing
    """
    # Ensure ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"
    
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Get all image files
    all_images = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append(os.path.join(root, file))
    
    # Shuffle the images
    random.shuffle(all_images)
    
    # Calculate split indices
    n_total = len(all_images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Split the data
    train_images = all_images[:n_train]
    val_images = all_images[n_train:n_train+n_val]
    test_images = all_images[n_train+n_val:]
    
    # Copy images to respective directories
    for img_path in train_images:
        filename = os.path.basename(img_path)
        shutil.copy(img_path, os.path.join(train_dir, filename))
    
    for img_path in val_images:
        filename = os.path.basename(img_path)
        shutil.copy(img_path, os.path.join(val_dir, filename))
    
    for img_path in test_images:
        filename = os.path.basename(img_path)
        shutil.copy(img_path, os.path.join(test_dir, filename))
    
    print(f"Dataset split complete:")
    print(f"  Train: {len(train_images)} images")
    print(f"  Validation: {len(val_images)} images")
    print(f"  Test: {len(test_images)} images")

if __name__ == "__main__":
    # Source directory containing MIT Indoor dataset
    source_dir = "/content/deepfillv2-pytorch/mit_indoor/Images"
    # Output directory for the split dataset
    output_dir = "/content/deepfillv2-pytorch/mit_indoor_split"
    
    split_dataset(source_dir, output_dir) 