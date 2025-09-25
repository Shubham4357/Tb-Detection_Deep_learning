
# Import Libraries
import os
import re
import cv2
import time
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    roc_auc_score,
    f1_score
)
from tqdm import tqdm
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from PIL import Image

# Define constants
IMG_TARGET_SIZE = (224, 224)  # Ukuran gambar input untuk DenseNet121
BATCH_SIZE = 32  # Ukuran batch untuk pelatihan
EPOCHS = 25  # Jumlah epoch untuk pelatihan
LEARNING_RATE = 0.0001

import os
import pandas as pd

# Directories for segmentation dataset
seg_dataset_path = '/kaggle/input/chest-x-ray-lungs-segmentation'
seg_images_path = os.path.join(seg_dataset_path, 'Chest-X-Ray', 'Chest-X-Ray', 'image')
seg_masks_path = os.path.join(seg_dataset_path, 'Chest-X-Ray', 'Chest-X-Ray', 'mask')

# Directories for classification dataset
cls_dataset_path = '/kaggle/input/tuberculosis-tb-chest-xray-dataset/TB_Chest_Radiography_Database'
cls_normal_path = os.path.join(cls_dataset_path, 'Normal')
cls_tb_path = os.path.join(cls_dataset_path, 'Tuberculosis')

# Count files for segmentation
seg_image_count = len(os.listdir(seg_images_path))
seg_mask_count = len(os.listdir(seg_masks_path))
seg_total_count = seg_image_count + seg_mask_count

# Count files for classification
cls_normal_count = len(os.listdir(cls_normal_path))
cls_tb_count = len(os.listdir(cls_tb_path))
cls_total_count = cls_normal_count + cls_tb_count

# Print counts
print("Segmentation - Image Count:", seg_image_count)
print("Segmentation - Mask Count:", seg_mask_count)
print("Segmentation - Total Files:", seg_total_count)
print("Classification - Normal Images:", cls_normal_count)
print("Classification - Tuberculosis Images:", cls_tb_count)
print("Classification - Total Images:", cls_total_count)

# Read segmentation metadata (if available)
seg_metadata_file = os.path.join(seg_dataset_path, 'MetaData.csv')
seg_metadata_df = pd.read_csv(seg_metadata_file) if os.path.exists(seg_metadata_file) else pd.DataFrame()

# Read classification metadata
cls_metadata_normal = pd.read_excel(os.path.join(cls_dataset_path, 'Normal.metadata.xlsx'))
cls_metadata_tb = pd.read_excel(os.path.join(cls_dataset_path, 'Tuberculosis.metadata.xlsx'))

# Combine classification metadata into one DataFrame
cls_metadata_df = pd.concat([
    cls_metadata_normal.assign(label=0), 
    cls_metadata_tb.assign(label=1)
], ignore_index=True)

# Display segmentation metadata
print("\nSegmentation Metadata - First 5 Rows:")
print(seg_metadata_df.head() if not seg_metadata_df.empty else "No segmentation metadata available")
print("\nSegmentation Metadata - Info:")
print(seg_metadata_df.info() if not seg_metadata_df.empty else "No segmentation metadata available")
print("\nSegmentation Metadata - Descriptive Stats:")
print(seg_metadata_df.describe())

# Display classification metadata
print("\nClassification Metadata - First 5 Rows:")
print(cls_metadata_df.head())
print("\nClassification Metadata - Info:")
print(cls_metadata_df.info())
print("\nClassification Metadata - Descriptive Stats:")
print(cls_metadata_df.describe())

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

# List image and mask files
seg_image_files = sorted(os.listdir(seg_images_path))
seg_mask_files = sorted(os.listdir(seg_masks_path))

# Function to display sample images and masks in two rows
def display_segmentation_samples(image_files, mask_files, img_dir, mask_dir, num_samples=5):
    plt.figure(figsize=(15, 10))
    
    # First row: original images
    for i in range(min(num_samples, len(image_files))):
        img_path = os.path.join(img_dir, image_files[i])
        img = Image.open(img_path)
        plt.subplot(2, num_samples, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Image {image_files[i]}\nSize: {img.size}")
        plt.axis('off')
    
    # Second row: corresponding masks
    for i in range(min(num_samples, len(mask_files))):
        mask_path = os.path.join(mask_dir, mask_files[i])
        mask = Image.open(mask_path)
        plt.subplot(2, num_samples, i+1 + num_samples)
        plt.imshow(mask, cmap='gray')
        plt.title(f"Mask {mask_files[i]}\nSize: {mask.size}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Show sample images and masks
print("Sample Segmentation Images and Masks:")
display_segmentation_samples(seg_image_files, seg_mask_files, seg_images_path, seg_masks_path)

# Data exploration: segmentation metadata statistics
print("\nCount per 'ptb' label in segmentation metadata:")
print(seg_metadata_df['ptb'].value_counts())

# Function to clean and validate image-mask pairs
def validate_image_mask_pairs(image_files, mask_files, img_dir, mask_dir):
    valid_pairs = []
    for img_file, mask_file in tqdm(zip(image_files, mask_files), total=len(image_files), desc="Validating Image-Mask Pairs"):
        img_path = os.path.join(img_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)
        try:
            with Image.open(img_path) as img, Image.open(mask_path) as mask:
                if img.size == mask.size and img.size[0] > 0 and img.size[1] > 0:
                    valid_pairs.append((img_file, mask_file))
        except Exception as e:
            print(f"Error loading {img_file} or {mask_file}: {e}")
    return valid_pairs

# Clean dataset
valid_pairs = validate_image_mask_pairs(seg_image_files, seg_mask_files, seg_images_path, seg_masks_path)
seg_image_files, seg_mask_files = zip(*valid_pairs)
print("Number of valid image-mask pairs:", len(seg_image_files))

# Function for preprocessing (resize + normalize)
def preprocess_segmentation_data(image_files, mask_files, img_dir, mask_dir, target_size=(256, 256)):
    processed_images = []
    processed_masks = []
    for img_file, mask_file in tqdm(zip(image_files, mask_files), total=len(image_files), desc="Resizing & Normalizing"):
        img_path = os.path.join(img_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)
        
        # Read in grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize
        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        
        # Normalize to 0–1 range
        img_normalized = img_resized / 255.0
        mask_normalized = mask_resized / 255.0
        
        processed_images.append(img_normalized)
        processed_masks.append(mask_normalized)
    
    return np.array(processed_images), np.array(processed_masks)

# Run preprocessing
IMG_TARGET_SIZE = (256, 256)
images_preprocessed, masks_preprocessed = preprocess_segmentation_data(
    seg_image_files, seg_mask_files, seg_images_path, seg_masks_path, IMG_TARGET_SIZE
)

# Ensure correct channel dimensions
if len(images_preprocessed.shape) == 3:
    images_preprocessed = np.expand_dims(images_preprocessed, axis=-1)
    masks_preprocessed = np.expand_dims(masks_preprocessed, axis=-1)

images_preprocessed = images_preprocessed.astype('float32')
masks_preprocessed = masks_preprocessed.astype('float32')

# Show examples after preprocessing
print("\nPreprocessed Samples:")
plt.figure(figsize=(10, 10))
for i in range(2):
    # Row 1: preprocessed image
    plt.subplot(2, 2, i+1)
    plt.imshow(images_preprocessed[i], cmap='gray')
    plt.title(f"Preprocessed Image {i+1}\nSize: {images_preprocessed[i].shape}")
    plt.axis('off')
    
    # Row 2: preprocessed mask
    plt.subplot(2, 2, i+1 + 2)
    plt.imshow(masks_preprocessed[i], cmap='gray')
    plt.title(f"Preprocessed Mask {i+1}\nSize: {masks_preprocessed[i].shape}")
    plt.axis('off')
plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Function to split data into Train/Validation/Test sets (70:15:15)
def split_dataset(images, masks, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    # First split: Train set and the remaining data
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, masks, train_size=train_ratio, random_state=random_state
    )
    
    # Second split: Validation and Test from remaining data
    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, train_size=val_test_ratio, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# Perform the split
X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(images_preprocessed, masks_preprocessed)
print("Dataset split completed.")

# Visualize label distribution with bar chart
subset_labels = ['Training', 'Validation', 'Test']
subset_sizes = [len(X_train), len(X_val), len(X_test)]
subset_colors = ['skyblue', 'lightgrey', 'salmon']  # Distinct colors for visibility

plt.figure(figsize=(8, 6))
bars = plt.bar(subset_labels, subset_sizes, color=subset_colors)
plt.title('Dataset Distribution (70:15:15)')
plt.xlabel('Subset')
plt.ylabel('Number of Samples')

# Add data labels above bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height, int(height), ha='center', va='bottom')
plt.show()

# Function to display example images & masks from each subset
def display_dataset_examples(X_train, X_val, X_test, y_train, y_val, y_test, num_samples=2):
    plt.figure(figsize=(15, 10))
    subsets = [
        (X_train, y_train, 'Training'),
        (X_val, y_val, 'Validation'),
        (X_test, y_test, 'Test')
    ]
    
    for i, (X_subset, y_subset, title) in enumerate(subsets):
        for j in range(min(num_samples, len(X_subset))):
            # Display image
            plt.subplot(len(subsets), num_samples * 2, j * 2 + 1 + i * num_samples * 2)
            plt.imshow(X_subset[j], cmap='gray')
            plt.title(f"{title} Image {j+1}\nSize: {X_subset[j].shape}")
            plt.axis('off')
            
            # Display mask
            plt.subplot(len(subsets), num_samples * 2, j * 2 + 2 + i * num_samples * 2)
            plt.imshow(y_subset[j], cmap='gray')
            plt.title(f"{title} Mask {j+1}\nSize: {y_subset[j].shape}")
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()

print("Sample images from each dataset subset:")
display_dataset_examples(X_train, X_val, X_test, y_train, y_val, y_test)

