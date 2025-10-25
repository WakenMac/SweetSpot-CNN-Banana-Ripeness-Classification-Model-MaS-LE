# Author: Waken Cean C. Maclang
# Date: October 18, 2025
# Course: Modelling and Simulation
# Task: Learning Evidence

# Split.py 
#     Splits the images into a 70-20-10 ratio for train, test, and validation datasets respectively.

# Works with Python 3.13.1

import os
import random
import shutil
from pathlib import Path

random.seed(777)

train_ratio = 0.8
validation_ratio = .10
test_ratio = .10

base_dir = Path('Datasets\\Annotated Dataset\\Combined Dataset')
output_dir = Path('Datasets\\GiMaTag Dataset')

# Ensures the output folders exists
for split in ['train', 'test', 'validate']:
    for class_name in os.listdir(base_dir):
        if class_name == 'Dataset Details.txt':
            continue
        # (f'{output_dir}\\{split}\\{class_name}').mkdir(parents=True, exist_ok = True)
        (output_dir / split / class_name).mkdir(parents=True, exist_ok = True)

# Splits the dataset
for class_name in os.listdir(base_dir):
    if class_name == 'Dataset Details.txt':
        continue

    class_dir = base_dir / class_name
    # Gets all files in the subtree (.jpg, .png, etc.)
    images_path = list(class_dir.glob("*.*"))
    # Shuffles the images
    random.shuffle(images_path)

    # Prepares index for splitting
    total = len(images_path)
    train_end = int(total * train_ratio)
    validation_end = train_end + int(total * validation_ratio)

    train_images = images_path[:train_end]
    validation_images = images_path[train_end:validation_end]
    test_images = images_path[validation_end:]

    # Copy Files
    for img in train_images:
        shutil.copy(img, output_dir / "train" / class_name / img.name)
    for img in validation_images:
        shutil.copy(img, output_dir / "validation" / class_name / img.name)
    for img in test_images:
        shutil.copy(img, output_dir / "test" / class_name / img.name)

print('Dataset successfully split into train, validation, and test folders!')