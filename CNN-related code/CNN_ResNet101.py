import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Load the ResNet101 model without its top classification layer
base_model = tf.keras.applications.ResNet101(
    include_top=False,         # remove the ImageNet classifier head
    weights='imagenet',        # load pretrained weights
    input_shape=(224, 224, 3)  # ResNet expects 224x224 RGB images
)

# 2. Freeze the base model so its weights are not updated during training
base_model.trainable = False

# 3. Build your own ANN classifier on top
model = models.Sequential([
    base_model,                                  # ResNet101 feature extractor
    layers.GlobalAveragePooling2D(),             # reduces spatial features
    layers.Dense(128, activation='relu'),        # your custom fully connected layer
    layers.Dropout(0.5),                         # helps prevent overfitting
    layers.Dense(4, activation='softmax')        # output: 4 ripeness categories
])

# 4. Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

#  Total params: 42,920,964 (163.73 MB)
#  Trainable params: 262,788 (1.00 MB)
#  Non-trainable params: 42,658,176 (162.73 MB)