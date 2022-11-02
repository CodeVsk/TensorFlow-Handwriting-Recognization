import tensorflow as tf
import cv2
import os
import numpy as np

class HandwritingRecognition:
    def __init__(self):
        self.path = os.getcwd();
        self.trainingPath = self.path+"/data/training";
        self.testingPath = self.path+"/data/testing";

    def import_datas(self):
        self.training = tf.keras.preprocessing.image.ImageDataGenerator(rescale= 1/255);
        self.validation = tf.keras.preprocessing.image.ImageDataGenerator(rescale= 1/255);

        self.trainingData = self.training.flow_from_directory(self.trainingPath, target_size=(128,128), batch_size=3, class_mode='binary');
        self.validationData = self.validation.flow_from_directory(self.testingPath, target_size=(128,128), batch_size=3, class_mode='binary');

    def create_model(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16,(3,3), activation="relu",input_shape=(128,128,3)),
            tf.keras.layers.MaxPool2D(2,2),

            tf.keras.layers.Conv2D(32,(3,3), activation="relu"),
            tf.keras.layers.MaxPool2D(2,2),

            tf.keras.layers.Conv2D(64,(3,3), activation="relu"),
            tf.keras.layers.MaxPool2D(2,2),

            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(512, activation='relu'),

            tf.keras.layers.Dense(1, activation='sigmoid')
        ]);

    def compile_model(self):
        self.model.compile(loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
            metrics=['accuracy']
        );

    def training_model(self):
        self.modelFit = self.model.fit(self.trainingData,
            steps_per_epoch=3,
            epochs=10,
            validation_data=self.validationData
        );

    def predict_image(self):
        for el in os.listdir("{}/testing".format(self.testingPath)):
            self.image = tf.keras.preprocessing.image.load_img("{}/testing/{}".format(self.testingPath, el));

            self.X = tf.keras.preprocessing.image.img_to_array(self.image);
            self.X = np.expand_dims(self.X, axis=0);

            self.imageTesting = np.vstack([self.X]);

            self.predict = self.model.predict(self.imageTesting);

            self.print_result();

    def print_result(self):
        if self.predict == 0:
            print("Imagem Analisada\nResultado: Letra 'i' localizada");
        else:
            print("Imagem Analisada\nResultado: Letra 'i' não localizada");


hr = HandwritingRecognition();
hr.import_datas();
hr.create_model();
hr.compile_model();
hr.training_model();
hr.predict_image();
