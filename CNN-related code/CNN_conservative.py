import tensorflow as tf
from tensorflow.keras import layers, models

# Adds noise to the data
# data_augmentation = tf.keras.Sequential([
#     layers.RandomFlip("horizontal"),
#     layers.RandomRotation(0.1),
#     layers.RandomZoom(0.1),
#     layers.RandomBrightness(0.1)
# ])

# Taken inspiration from the VGGNet (2014) Architecture consisting of 3x3 filters but with smaller filters
model = models.Sequential([
    # data_augmentation,
    layers.Input(shape=(224, 224, 3)),

    # --- Block 1 ---
    layers.Conv2D(16, (3, 3), activation=None, padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Conv2D(16, (3, 3), activation=None, padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # --- Block 2 ---
    layers.Conv2D(32, (3, 3), activation=None, padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Conv2D(32, (3, 3), activation=None, padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # --- Block 3 ---
    layers.Conv2D(64, (3, 3), activation=None, padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.4),

    # Imagine all of the Convolutional and Pooling layers are above
    # Then we connect Allan's ANN design here

    # --- Fully Connected Layers ---
    # We will plug your designed ANN here :>
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')  # Assuming 4 ripeness categories
])

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

#  Total params: 3,247,380 (12.39 MB)
#  Trainable params: 3,247,060 (12.39 MB)
#  Non-trainable params: 320 (1.25 KB)