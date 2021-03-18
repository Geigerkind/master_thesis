from tensorflow import keras
from tensorflow.keras import layers


class GenerateFFNN:
    def __init__(self):
        # Configuration
        self.input_size = 2
        self.intermediate_size = 3
        self.output_size = 2

        # Train the model
        self.keras_model = self.model()
        # TODO: Train

    def predict(self):
        raise Exception("Not implemented.")

    def model(self):
        return keras.Sequential([
            layers.Dense(self.input_size, activation="relu"),
            layers.Dense(self.intermediate_size, activation="softmax"),
            layers.Dense(self.output_size),
        ])
