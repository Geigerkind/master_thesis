import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


class GenerateFFNN:
    def __init__(self):
        # Configuration
        self.input_size = 36
        self.intermediate_size = 100
        self.output_size = 9

        # Train the model
        self.keras_model = self.model()
        self.keras_model.compile(optimizer='adam', loss='categorical_crossentropy')
        # self.keras_model.summary()

    def fit(self, training_data_x, training_data_y):
        self.keras_model.fit(np.asarray(training_data_x), np.asarray(training_data_y), batch_size=50, epochs=100,
                             verbose=0)

    def predict(self, data):
        return self.keras_model.predict(np.asarray(data))

    def model(self):
        return keras.Sequential([
            layers.Dense(self.input_size, input_dim=self.input_size, activation="relu"),
            layers.Dense(self.intermediate_size, activation="relu"),
            layers.Dense(self.output_size, activation="sigmoid"),
        ])

    def evaluate_accuracy(self, prediction, reality):
        correct = 0
        for i in range(len(prediction)):
            if np.array(prediction[i]).argmax() == np.array(reality[i]).argmax():
                correct = correct + 1

        return correct / len(prediction)
