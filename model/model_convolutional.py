import os

import tensorflow as tf
from keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Rescaling, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.preprocessing import image_dataset_from_directory

import matplotlib.pyplot as plt
import random


way = 'C:\\Users\\stask\\PycharmProjects\\Kursovstas\\model\data\\archive_ds\\dataset'
train_dir = os.path.join(way, 'train')


train_dataset = image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    labels='inferred',
    label_mode='int',
    image_size=(160, 160),
    batch_size=32,
    shuffle=True
)


model = Sequential([
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.2),
    RandomZoom(0.1),
    Rescaling(1./255),
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    MaxPooling2D((2, 2), strides=2),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    MaxPooling2D((2, 2), strides=2),
    Flatten(),
    Dense(5, activation='softmax')
])


base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])


model.fit(train_dataset, epochs=10, )

model.save('convolutional.keras')