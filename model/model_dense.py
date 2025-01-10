import os

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Rescaling, RandomFlip, RandomRotation, RandomZoom
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
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.2),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(5, activation='softmax')
])


base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])


model.fit(train_dataset, epochs=5)
model.summary()
model.save('dense.keras')