# Author: Waken Cean C. Maclang
# Date: October 16, 2025
# Course: Modelling and Simulation
# Task: Learning Evidence

# Resize.py 
#     Resizes images found at the "source_dir" and saves it at the "target_dir".

# Works with Python 3.13.1

import os
import cv2

# source_dir = "Datasets\\Original Datasets\\Shahriar Dataset\Rotten"
# target_dir = "Datasets\\Annotated Dataset\\Shahriar Dataset\\Rotten"

# source_dir = "Datasets\\Original Datasets\\Shahriar Dataset\\Unripe"
# target_dir = "Datasets\\Annotated Dataset\\Shahriar Dataset\\Unripe"

# source_dir = "Datasets\\Original Datasets\\Shahriar Dataset\\Ripe"
# target_dir = "Datasets\\Annotated Dataset\\Shahriar Dataset\\Ripe"

# source_dir = "Datasets\\Original Datasets\\Shahriar Dataset\\Overripe"
# target_dir = "Annotated Dataset\\Shahriar Dataset\\Overripe"

# source_dir = "Datasets\\Original Datasets\\Fayoum_University_Banana _Classes\\Overripe"
# target_dir = "Datasets\\Annotated Dataset\\Fayoum_University_Banana _Classes\\Overripe"

# source_dir = "Datasets\\Original Datasets\\Fayoum_University_Banana _Classes\\Ripe"
# target_dir = "Datasets\\Annotated Dataset\\Fayoum_University_Banana _Classes\\Ripe"

ref_source_dir = "Datasets\\Original Datasets\\Fayoum_University_Banana _Classes\\"
ref_target_dir = "Datasets\\Annotated Datasets\\Fayoum_University_Banana _Classes\\"

index = 1
for class_name in ['Green', 'Midripen', 'Overripen', 'Yellowish_Green']:
    source_dir = ref_source_dir + class_name
    target_dir = ref_target_dir + class_name
    for filename in os.listdir(source_dir):
        if filename.lower().endswith(('.jpg', '.png', 'jpeg')):

            # Get the filename and the file extension
            name, ext = os.path.splitext(filename)

            # Creates the new name for our image
            new_name = ""
            if source_dir.lower().__contains__('shahriar dataset'):
                part_name = name.split('-')
                for i in range(3):
                    new_name += part_name[i] + '-'
                new_name += str(index)
            elif source_dir.lower().__contains__('fayoum_university_banana _classes'):
                # new_name += 'plantain'
                
                # if source_dir.lower().endswith('overripe'):
                #     new_name += '-overripe'
                # elif source_dir.lower().endswith('unripe'):
                #     new_name += '-unripe'
                # elif source_dir.lower().endswith('ripe'):
                #     new_name += '-ripe'
                
                new_name += 'plantain' + '-' +  class_name.lower() + '-' + name.split('.')[0][1:]

                # new_name += '-' + str(f'{index:0}')
            else:
                print(f'Skipping the image: {name}.')
                continue

            # Checks if the image already exists in the target directory
            if cv2.imread(os.path.join(target_dir, f'{new_name}-annotated{ext}')) is not None:
                print(f'Skipping image {filename} as it already exists in the target directory: {new_name}{ext}')
                index += 1
                continue

            # Gets the image
            image_path = os.path.join(source_dir, filename)
            image = cv2.imread(image_path)

            # Check if image exists
            if image is None:
                print(f'Skipping {filename}: could not load.')
                continue
            
            # Resize the image 
            resized = cv2.resize(image, (224, 224))

            # Saves the image to the target location
            new_file_name = f'{new_name}-annotated{ext}'
            new_path = os.path.join(target_dir, new_file_name)
            cv2.imwrite(new_path, resized)

            print(f'Saved: {new_path}')
            index += 1
