from tensorflow import keras
from tensorflow.keras import layers


class GenerateFFNN:
    def __init__(self, training_data_x, training_data_y):
        self.training_data_x = training_data_x
        self.training_data_y = training_data_y

        # Configuration
        self.input_size = 4
        self.intermediate_size = 30
        self.output_size = 1

        # Train the model
        self.keras_model = self.model()
        self.keras_model.compile(optimizer='adam', loss='binary_crossentropy')
        self.keras_model.fit(self.training_data_x, self.training_data_y, batch_size=50, epochs=30)
        self.keras_model.summary()

    def predict(self, data):
        return self.keras_model.predict(data)

    def model(self):
        return keras.Sequential([
            layers.BatchNormalization(input_dim=4),
            layers.Dense(self.input_size, input_dim=4, activation="relu"),
            layers.Dense(self.intermediate_size, activation="relu"),
            layers.Dense(self.intermediate_size, activation="relu"),
            layers.Dense(self.intermediate_size, activation="relu"),
            layers.Dense(self.output_size, activation="sigmoid"),
        ])