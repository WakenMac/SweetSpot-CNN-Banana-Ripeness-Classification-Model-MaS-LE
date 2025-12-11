# Author: Waken Cean C. Maclang
# Date: October 18, 2025
# Course: Modelling and Simulation
# Task: Learning Evidence

# Split.py 
#     Splits the images into a 70-20-10 ratio for train, test, and validation datasets respectively.

# Works with Python 3.13.1

import os
import random
from pathlib import Path

random.seed(777)

train_ratio = 0.5
validation_ratio = .25
test_ratio = .25

base_dir = Path('Datasets\\Annotated Datasets\\Fayoum_University_Banana _Classes')
# output_dir = Path('Datasets\\Annotated Datasets\\Augment')
output_dir = Path('Datasets\\Annotated Datasets\\Split')

# Ensures the output folders exists
for split in ['train', 'test', 'validation']:
    for class_name in os.listdir(base_dir):
        if class_name == 'Dataset Details.txt':
            continue
        # (f'{output_dir}\\{split}\\{class_name}').mkdir(parents=True, exist_ok = True)
        (output_dir / split / class_name).mkdir(parents=True, exist_ok = True)

# Splits the dataset
train, test, val = [0, 0, 0]
for i, class_name in enumerate(os.listdir(base_dir)):
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

    multipliers = [1, 1.2, 3.2, 2.2]
    train += int(train_end * 3 * multipliers[i])
    val += validation_end - train_end
    test += len(images_path) - validation_end
    print(f'{class_name}: {int(train_end * 3 * multipliers[i])} | {len(images_path) - validation_end} | {validation_end - train_end}')

    # Copy Files
    # for img in train_images:
    #     shutil.copy(img, output_dir / "train" / class_name / img.name)
    # for img in validation_images:
    #     shutil.copy(img, output_dir / "validation" / class_name / img.name)
    # for img in test_images:
    #     shutil.copy(img, output_dir / "test" / class_name / img.name)

print(train, val, test, sep=' | ')
print('Dataset successfully split into train, validation, and test folders!')