# Author: Waken Cean C. Maclang
# Date: October 16, 2025
# Course: Modelling and Simulation
# Task: Learning Evidence

# Augment.py 
#     Creates augmented versions of images found at the "source_dir" and saves it at the "target_dir".

# Works with Python 3.13.1

import os
import cv2
import random
import numpy as np
import pandas as pd

# source_dir = "Datasets\\Annotated Dataset\\Fayoum Uni Dataset\\Overripe"
# target_dir = "Datasets\\Annotated Dataset\\Fayoum Uni Augmented Dataset\\Overripe"

# source_dir = "Datasets\\Annotated Dataset\\Fayoum Uni Dataset\\Ripe"
# target_dir = "Datasets\\Annotated Dataset\\Fayoum Uni Augmented Dataset\\Ripe"

ref_source_dir = "Datasets\\Annotated Datasets\\Fayoum_University_Banana _Classes\\"
ref_target_dir = "Datasets\\Annotated Datasets\\Augment\\train\\"

# Helper Methods
def rotate_image(image, angle):
    """Rotates the image by the given angle (in degrees)"""
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    return cv2.warpAffine(image, matrix, (w, h))

def flip_image(image, mode):
    """Flips an image vertically or horizontally."""
    return cv2.flip(src=image, flipCode=mode)

def crop_zoom(image, zoom_percent):
    """Crop the image and zooms back to its original size."""
    h, w = image.shape[:2]
    zoom = zoom_percent / 100.0
    crop_h = int(h * zoom)
    crop_w = int(w * zoom)
    y1 = crop_h // 2
    x1 = crop_w // 2
    cropped = image[y1:h - y1, x1:w - x1]  # How does this work?
    return cv2.resize(cropped, (w, h))

def blur_image(image, strength = 1):
    """Applies Gaussian blur up to 1px."""
    return cv2.GaussianBlur(image, (2 * strength + 1, 2 * strength + 1), 0)

# Will not be used as it may affect the color of the banana (causing errors in classification)
# def adjust_hsv(image, hue = 0, sat = 0, val = 0):
#     """Adjust hue, saturation, and brightness (value)."""
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
#     h, s, v = cv2.split(hsv)
#     h = (h + hue) % 180
#     s = np.minimum(255, np.maximum(0, s * (1 + sat / 100.0)))
#     v = np.minimum(255, np.maximum(0, v * (1 + val / 100.0)))
#     hsv_merged = cv2.merge([h, s, v])
#     hsv_merged = hsv_merged.astype(np.uint8)
#     return cv2.cvtColor(hsv_merged, cv2.COLOR_HSV2BGR)

def add_gaussian_noise(image, mean=0, std=15):
    image = image.astype(np.float32)
    noise = np.empty(image.shape, dtype=np.float32)
    cv2.randn(noise, mean, std)
    noisy_image = image.astype(np.float32) + noise
    noisy_image = np.minimum(255, np.maximum(0, noisy_image)).astype(np.uint8)
    return noisy_image

# ==================================================================================

# Main for loop
random.seed(777)
DATASET_PATH = 'augmented_data.csv'
dataset = None
if os.path.exists(DATASET_PATH):
    dataset = pd.read_csv(DATASET_PATH)
else:
    dataset = pd.DataFrame()

# for i, class_name in enumerate(['Green', 'Midripen', 'Overripen', 'Yellowish_Green']):
for i, class_name in enumerate(['Overripe', 'Ripe', 'Unripe']):
    source_dir = ref_source_dir + class_name
    target_dir = ref_target_dir + class_name

    for filename in os.listdir(source_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):

            name, ext = os.path.splitext(filename)
            image_path = os.path.join(source_dir, filename)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Skipping {filename}: could not load.")
                continue
            
            # Creates the new name (plantain-overripe-0-augmented-index) 
            # from the old name (plantain-overripe-0-annotated.jpg)
            new_name = ""
            parts = name.split('-')[0:3]
            for part in parts:
                new_name += part + '-'
            new_name += 'augmented-'

            # TODO: Make a new algorithm to create unique augmented images
            
            augmented_copies = 10
            multipliers = [1, 1.2, 3.2, 2.2]
            # for j in range(int(augmented_copies * multipliers[i])):  # Generate 10 random augmentations per image
            for j in range(augmented_copies):                  
                std = random.randint(0, 50)
                blur = random.randint(0, 1)
                angle = random.uniform(-30, 30)
                zoom = random.randint(0, 20)
                mode = random.randint(-1, 1)

                img_aug = add_gaussian_noise(image, std=std)
                img_aug = blur_image(img_aug, blur)
                img_aug = rotate_image(img_aug, angle)
                img_aug = crop_zoom(img_aug, zoom)
                img_aug = flip_image(img_aug, mode)
                # cv2.imwrite(os.path.join(target_dir, f"{new_name}{j}{ext}"), img_aug)

                # Find a way to store the augmentations into a DataFrame, then into a CSV for safekeeping
                my_dict = {
                    'type':[class_name],
                    'original-index':[parts[2]],
                    'augmentation-index':[j],
                    'gaussian_noise':[std],
                    'blur':[blur],
                    'rotation_angle':[angle],
                    'zoom':[zoom],
                    'flip_image':['vertical' if mode == 0 else 'horizontal' if mode == 1 else 'both']
                }
                tempDF = pd.DataFrame(my_dict)
                dataset = pd.concat([dataset, tempDF], axis=0)
            
            print(f"Augmented {filename}")

dataset.to_csv(DATASET_PATH, index=False)

# Count the number of files

ref_source_dir = "Datasets\\Annotated Datasets\\Split\\"
ref_target_dir = "Datasets\\Old GiMaTag Dataset\\"
for i, folder_name in enumerate(['train', 'test', 'validation']):
    main_target_dir = ref_target_dir + folder_name 
    size = 0
    for i, class_name in enumerate(['Overripe', 'Ripe', 'Rotten', 'Unripe']):
        target_dir = main_target_dir + '\\' + class_name
        temp_size = len(os.listdir(target_dir))
        size += temp_size

        print(f'{class_name} # of images: {temp_size}')
    print(f'Total # of {folder_name} images: {size}')
    print('\n\n')
