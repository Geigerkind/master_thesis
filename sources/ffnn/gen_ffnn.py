import copy
import os
import random
from multiprocessing import Pool

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sources.config import NUM_CORES


def do_work(args):
    copy_test_features, num_outputs, continued_predict, test_labels = args
    fitted_model = keras.models.load_model(
        "/home/shino/Uni/master_thesis/external_eval/bin/eval_1_DT_16_32_KNN_1_64_75_DS_1234/evaluation_knn_model.h5",
        compile=False)
    # Calculate accuracy
    ctf_predictions = GenerateFFNN.static_continued_predict(fitted_model, copy_test_features,
                                                            num_outputs) if continued_predict else fitted_model.predict(
        copy_test_features)
    ctf_accuracy = GenerateFFNN.internal_evaluate_accuracy(ctf_predictions, test_labels)

    return ctf_accuracy


class GenerateFFNN:
    def __init__(self, input_size, output_size, num_hidden_layers, nodes_hidden_layer, num_epochs, is_binary=False):
        """
        Generates an FFNN with one hidden layer.
        It uses ReLU and SoftMax for the last layer.

        :param input_size: Length of the feature sets
        :param output_size: Amount of classes to be predicted
        :param num_hidden_layers: The number of hidden layers in the model structure
        :param nodes_hidden_layer: Amount of neurons per hidden layer
        :param num_epochs: Number of epochs to be trained
        :param is_binary: Uses improved loss function and sigmoid in the output then
        """
        self.history = 0

        # Set random seeds for reproducible results
        os.environ['PYTHONHASHSEED'] = str(0)
        random.seed(0)
        # tf.random.set_seed(0)
        session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=NUM_CORES,
                                                inter_op_parallelism_threads=NUM_CORES)
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
        tf.compat.v1.keras.backend.set_session(sess)

        # Configuration
        self.input_size = input_size
        self.intermediate_size = nodes_hidden_layer
        self.output_size = output_size
        self.num_hidden_layers = num_hidden_layers
        self.num_epochs = num_epochs
        self.batch_size = 50
        self.is_binary = is_binary

        # Train the model
        self.keras_model = self.model()
        if self.is_binary:
            self.keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["acc"])
        else:
            self.keras_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["acc"])
        # self.keras_model.summary()

    def fit(self, training_data_x, training_data_y, validation_data_x, validation_data_y):
        """
        Uses the keras model fit function on the compiled model.
        """
        self.history = self.keras_model.fit(np.asarray(training_data_x), np.asarray(training_data_y),
                                            batch_size=self.batch_size, epochs=self.num_epochs, verbose=0,
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
        return GenerateFFNN.static_continued_predict(self, data, self.output_size)

    @staticmethod
    def static_continued_predict(model, data, output_size):
        """
        Same as continued_predict, just static for evaluation purposes.
        """
        # Assumes that feature 0 and 1 are previous locations
        data_copy = np.asarray(data).copy()
        # data_copy = data
        data_copy[0][0] = 0
        data_copy[0][1] = 0
        predictions = []
        data_copy_len = len(data_copy)
        prediction = model.predict_on_batch(np.asarray([data_copy[0]]))[0]
        predictions.append(prediction)
        last_distinct_locations = [0, 0]
        for i in range(1, data_copy_len):
            prediction = model.predict_on_batch(data_copy[i:i + 1])[0]
            if i < data_copy_len - 1:
                predicted_location = np.asarray(prediction).argmax()
                if 0 < predicted_location != last_distinct_locations[-1]:
                    last_distinct_locations.append(predicted_location)
                    last_distinct_locations.pop(0)

                data_copy[i + 1][0] = predicted_location / (output_size - 1)
                if predicted_location == 0:
                    data_copy[i + 1][1] = last_distinct_locations[-1] / (output_size - 1)
                else:
                    data_copy[i + 1][1] = last_distinct_locations[-2] / (output_size - 1)
            predictions.append(prediction)
        return predictions

    def model(self):
        """
        Specifies the model layout.

        :return: Returns the model template.
        """
        model = keras.Sequential()
        model.add(layers.Dense(self.input_size, input_dim=self.input_size, activation="relu"))
        for _ in range(self.num_hidden_layers):
            model.add(layers.Dense(self.intermediate_size, activation="relu"))
        if self.is_binary:
            model.add(layers.Dense(1, activation="sigmoid"))
        else:
            model.add(layers.Dense(self.output_size, activation="softmax"))
        return model

    def evaluate_accuracy(self, prediction, reality):
        return self.__evaluate_accuracy(prediction, reality, self.is_binary)

    @staticmethod
    def internal_evaluate_accuracy(prediction, reality, is_binary=False):
        return GenerateFFNN.__evaluate_accuracy(prediction, reality, is_binary)

    @staticmethod
    def __evaluate_accuracy(prediction, reality, is_binary=False):
        """
        Compares predicted data and actual data.

        :param prediction: Array of predictions
        :param reality: Array of actual labels
        :param is_binary: If the model is a binary model
        :return: Accuracy (float)
        """

        correct = 0
        if is_binary:
            for i in range(len(prediction)):
                pred_label = 1 if prediction[i][0] >= 0.5 else 0
                if pred_label == reality[i]:
                    correct = correct + 1
        else:
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

    @staticmethod
    def feature_importances(test_features, test_labels, num_outputs, continued_predict=True,
                            is_binary=False, NUM_CORES=1):
        """
        In comparison to Decision Trees, Neural Networks dont provide such a function.
        However there is something called "Permutation Importance".
        This was proposed by Leo Breiman in the Random Forest paper.
        It calculates the accuracy for a provided data set.
        We then shuffle a feature and see the error we get compared to the correct order.
        The higher the error, the more important the feature is.

        :param fitted_model: The fitted FFNN
        :param test_features: The feature sets to infer the importance on
        :param test_labels: Labels for the feature sets
        :return: Array of errors for each feature, higher is more important
        """

        test_accuracy = Pool(1).map(do_work, [[np.asarray(test_features), num_outputs, continued_predict, test_labels]])[0]

        map_args = []
        test_len = len(test_features)
        for i in range(len(test_features[0])):
            # Shuffle column i
            permutation = np.random.permutation(test_len)
            copy_test_features = np.asarray(copy.deepcopy(test_features))
            for ctf_index in range(test_len):
                copy_test_features[ctf_index][i] = test_features[permutation[ctf_index]][i]
            map_args.append(
                [copy_test_features, num_outputs, continued_predict, test_labels])

        return [test_accuracy - x for x in Pool(NUM_CORES).map(do_work, map_args)]
