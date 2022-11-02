#from tensorflow.keras.preprocessing.image import imageDataGenerator
#from tensorflow.keras.preprocessing import image
#from tensorflow.keras.optimizers import RMSprop

import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np


path_training = "E:/Desenvolvimento/Deep Learning/version-2.0/data/training"
path_testing = "E:/Desenvolvimento/Deep Learning/version-2.0/data/testing"

img = tf.keras.preprocessing.image.load_img("./data/training/i/train_69_00001.png")

image = cv2.imread("./data/training/i/train_69_00001.png").shape

training = tf.keras.preprocessing.image.ImageDataGenerator(rescale= 1/255)

validation = tf.keras.preprocessing.image.ImageDataGenerator(rescale= 1/255)

training_data = training.flow_from_directory("E:/Desenvolvimento/Deep Learning/version-2.0/data/training", target_size=(128,128),batch_size=3,class_mode='binary')

validation_data = validation.flow_from_directory("data/testing/", target_size=(128,128),batch_size=3,class_mode='binary')

print(training_data.class_indices)
print(training_data.classes)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3), activation="relu",input_shape=(128,128,3)),
    tf.keras.layers.MaxPool2D(2,2),

    tf.keras.layers.Conv2D(32,(3,3), activation="relu"),
    tf.keras.layers.MaxPool2D(2,2),

    tf.keras.layers.Conv2D(64,(3,3), activation="relu"),
    tf.keras.layers.MaxPool2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
    metrics=['accuracy']
)

model_fit = model.fit(training_data,
    steps_per_epoch=3,
    epochs=10,
    validation_data=validation_data
)

for i in os.listdir(path_testing+"/i"):
    img_testing = tf.keras.preprocessing.image.load_img(path_testing+"/i/"+i)

    X = tf.keras.preprocessing.image.img_to_array(img_testing)
    X = np.expand_dims(X, axis=0)

    img_testings = np.vstack([X])

    predict = model.predict(img_testings)

    if predict == 0:
        print("""

Imagem Analisada
Resultado: Letra 'i' localizada

        """
        )
    else:
        print("""
        
Imagem Analisada
Resultado: Letra 'i' não localizada

        """
        )