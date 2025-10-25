# Creating our own Banana Classifier

# Things to consider:
#     Normalize pixel values
#     One hot encoding for the 4 label classes
#     Use early stopping to prevent overfitting
#     Split the data to (70-20-10) train, val, test

import tensorflow as tf
from tensorflow.keras import layers, models

# Adds noise to the data
# data_augmentation = tf.keras.Sequential([
#     layers.RandomFlip("horizontal"),
#     layers.RandomRotation(0.1),
#     layers.RandomZoom(0.1),
#     layers.RandomBrightness(0.1)
# ])

# Example for 64x64 RGB images with 4 classes
# Taken inspiration from LeNet architecture and VGGNet filters
vanilla_model = models.Sequential([
    # data_augmentation,

    # Convolution + Pooling Block 1
    # Improve the image size
    layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(64, 64, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    # Block 2
    layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    # Block 3
    layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    # Block 4
    layers.Conv2D(256, (3,3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    # Flatten + Dense Layers
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),  # helps prevent overfitting

    # Output layer
    layers.Dense(4, activation='softmax')  # 4 classes
])

# Compile model
vanilla_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

vanilla_model.summary()

#  Total params: 1,440,196 (5.49 MB)
#  Trainable params: 1,439,236 (5.49 MB)
#  Non-trainable params: 960 (3.75 KB)

# Training the model
# history = model.fit(
#     train_dataset,
#     validation_data=val_dataset,
#     epochs=30
# )

# # Evaluating the generalization of the model
# test_loss, test_acc = model.evaluate(test_dataset)
# print(f"Test accuracy: {test_acc:.2f}")