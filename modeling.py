import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# dimensions and batch size
height, width = 180, 180
batch = 32

# load organized car training data
train_ds = tf.keras.utils.image_dataset_from_directory(
    "data/cars_train_organized",
    image_size=(height, width),
    batch_size=batch,
    shuffle=True
)

class_names = train_ds.class_names

# normalize
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

# CNN model
model = keras.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(height, width, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names))  # 196 classes
])

# compile the model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# train the model
model.fit(train_ds, epochs=10)

# save the model
model.save("car_model.keras")

# test the model
img = tf.keras.utils.load_img("data/cars_test/00001.jpg", 
                           target_size=(height, width))

img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
"This image most likely belongs to {} with a {:.2f} percent confidence."
.format(class_names[np.argmax(score)], 100 * np.max(score))
)