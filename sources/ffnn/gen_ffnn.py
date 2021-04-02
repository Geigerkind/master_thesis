import os
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class GenerateFFNN:
    def __init__(self, input_size, output_size):
        self.history = 0

        # Set random seeds for reproducible results
        os.environ['PYTHONHASHSEED'] = str(0)
        random.seed(0)
        tf.random.set_seed(0)
        session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=16, inter_op_parallelism_threads=16)
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
        tf.compat.v1.keras.backend.set_session(sess)

        # Configuration
        self.input_size = input_size
        self.intermediate_size = 30
        self.output_size = output_size

        # Train the model
        self.keras_model = self.model()
        self.keras_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
        # self.keras_model.summary()

    def fit(self, training_data_x, training_data_y, validation_data_x, validation_data_y):
        self.history = self.keras_model.fit(np.asarray(training_data_x), np.asarray(training_data_y), batch_size=50,
                                            epochs=50, verbose=0,
                                            validation_data=(np.asarray(validation_data_x), np.asarray(validation_data_y)))

    def predict(self, data):
        return self.keras_model.predict(np.asarray(data))

    def model(self):
        return keras.Sequential([
            layers.Dense(self.input_size, input_dim=self.input_size, activation="relu"),
            layers.Dense(self.intermediate_size, activation="relu"),
            layers.Dense(self.intermediate_size, activation="relu"),
            layers.Dense(self.intermediate_size, activation="relu"),
            layers.Dense(self.output_size, activation="sigmoid"),
        ])

    def evaluate_accuracy(self, prediction, reality):
        correct = 0
        for i in range(len(prediction)):
            if np.array(prediction[i]).argmax() == np.array(reality[i]).argmax():
                correct = correct + 1

        return correct / len(prediction)

    def get_history(self):
        return self.history.history
