import tensorflow as tf
from PIL import Image
import os

# Image resizing
input_folder = "datasets/raw_images/"
output_folder = "datasets/resized_images/"
target_size = (64, 64)  # matches your CNN input

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        img = Image.open(os.path.join(input_folder, filename))
        img = img.resize(target_size)
        img.save(os.path.join(output_folder, filename))

# Importing the data and preparing the dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    'Datasets/Final Dataset/train', image_size=(224, 224), batch_size=32)

val_ds = tf.keras.utils.image_dataset_from_directory(
    'Datasets/Final Dataset/validation', image_size=(224, 224), batch_size=32)

test_ds = tf.keras.utils.image_dataset_from_directory(
    'Datasets/Final Dataset/test', image_size=(224, 224), batch_size=32)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",
    image_size=(128, 128),   # same as your model input
    batch_size=32,
    label_mode='categorical' # because you’re using softmax (4 classes)
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset/validation",
    image_size=(128, 128),
    batch_size=32,
    label_mode='categorical'
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size=(128, 128),
    batch_size=32,
    label_mode='categorical'
)

# Normalization of pixel data
normalization_layer = tf.keras.layers.Rescaling(1./255)

train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset   = val_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset  = test_dataset.map(lambda x, y: (normalization_layer(x), y))

# Optimize Data Loading
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset   = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset  = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)