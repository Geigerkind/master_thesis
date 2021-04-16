import os
import random
from multiprocessing import cpu_count

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class GenerateFFNN:
    def __init__(self, input_size, output_size):
        """
        Generates an FFNN with one hidden layer.
        It uses ReLU and SoftMax for the last layer.

        :param input_size: Length of the feature sets
        :param output_size: Amount of classes to be predicted
        """
        self.history = 0

        # Set random seeds for reproducible results
        os.environ['PYTHONHASHSEED'] = str(0)
        random.seed(0)
        tf.random.set_seed(0)
        session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=cpu_count(),
                                                inter_op_parallelism_threads=cpu_count())
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
        tf.compat.v1.keras.backend.set_session(sess)

        # Configuration
        self.input_size = input_size
        self.intermediate_size = 60
        self.output_size = output_size

        # Train the model
        self.keras_model = self.model()
        self.keras_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
        # self.keras_model.summary()

    def fit(self, training_data_x, training_data_y, validation_data_x, validation_data_y):
        """
        Uses the keras model fit function on the compiled model.
        """
        self.history = self.keras_model.fit(np.asarray(training_data_x), np.asarray(training_data_y), batch_size=50,
                                            epochs=75, verbose=0,
                                            validation_data=(
                                            np.asarray(validation_data_x), np.asarray(validation_data_y)))

    def predict(self, data):
        """
        Uses the predict function of the keras model.
        """
        return self.keras_model.predict(np.asarray(data))

    def continued_predict(self, data):
        """
        Iteratively predicts the location of the provided data.
        For through explanation see GenerateDecisionTree or thesis.
        """
        # Assumes that feature 0 and 1 are previous locations
        data_copy = np.asarray(data).copy()
        predictions = []
        data_copy_len = len(data_copy)
        prediction = self.predict([data_copy[0]])[0]
        predictions.append(prediction)
        prev_predicted_location = np.asarray(prediction).argmax() / self.output_size
        last_distinct_location = 0
        for i in range(1, data_copy_len):
            prediction = self.predict([data_copy[i]])[0]
            if i < data_copy_len - 1:
                predicted_location = np.asarray(prediction).argmax() / self.output_size
                if predicted_location != prev_predicted_location and prev_predicted_location != last_distinct_location \
                        and prev_predicted_location > 0:
                    last_distinct_location = prev_predicted_location

                data_copy[i + 1][0] = predicted_location
                data_copy[i + 1][1] = last_distinct_location
                prev_predicted_location = predicted_location
            predictions.append(prediction)
        return predictions

    @staticmethod
    def static_continued_predict(model, data, output_size):
        """
        Same as continued_predict, just static for evaluation purposes.
        """
        # Assumes that feature 0 and 1 are previous locations
        data_copy = np.asarray(data).copy()
        predictions = []
        data_copy_len = len(data_copy)
        prediction = model.predict(np.asarray([data_copy[0]]))[0]
        predictions.append(prediction)
        prev_predicted_location = np.asarray(prediction).argmax() / output_size
        last_distinct_location = 0
        for i in range(1, data_copy_len):
            prediction = model.predict(np.asarray([data_copy[i]]))[0]
            if i < data_copy_len - 1:
                predicted_location = np.asarray(prediction).argmax() / output_size
                if predicted_location != prev_predicted_location and prev_predicted_location != last_distinct_location \
                        and prev_predicted_location > 0:
                    last_distinct_location = prev_predicted_location

                data_copy[i + 1][0] = predicted_location
                data_copy[i + 1][1] = last_distinct_location
                prev_predicted_location = predicted_location
            predictions.append(prediction)
        return predictions

    def model(self):
        """
        Specifies the model layout.

        :return: Returns the model template.
        """
        return keras.Sequential([
            layers.Dense(self.input_size, input_dim=self.input_size, activation="relu"),
            layers.Dense(self.intermediate_size, activation="relu"),
            layers.Dense(self.output_size, activation="sigmoid"),
        ])

    def evaluate_accuracy(self, prediction, reality):
        """
        Compares predicted data and actual data.

        :param prediction: Array of predictions
        :param reality: Array of actual labels
        :return: Accuracy (float)
        """

        correct = 0
        for i in range(len(prediction)):
            if np.array(prediction[i]).argmax() == np.array(reality[i]).argmax():
                correct = correct + 1

        return correct / len(prediction)

    def get_history(self):
        """
        :return: keras history of the fitting process
        """
        return self.history.history

    def save(self, file_path):
        """
        Saves the model to the specified path.
        """
        self.keras_model.save(file_path)
