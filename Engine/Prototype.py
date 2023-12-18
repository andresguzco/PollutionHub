import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from typing import Tuple

# TODO: Set up NN for the linear regression parameter


class CustomModel:
    def __init__(self, input_size: int = 100000) -> None:
        self.input_size: int = input_size
        self.model: tf.keras.Sequential = self._build_model()

    def _generate_data(self) -> Tuple[np.ndarray, np.ndarray]:
        xTrain = np.sort(np.random.uniform(-1, 1, size=self.input_size)).reshape(-1, 1)
        yTrain = 2 + np.cos(4 * np.pi * xTrain)
        return xTrain, yTrain

    @staticmethod
    def _build_model() -> tf.keras.Sequential:
        Model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        return Model

    def train(self, epochs: int = 100, batch_size: int = 256) -> None:
        xTrain, yTrain = self._generate_data()
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                           loss='mean_squared_error')
        self.model.fit(xTrain, yTrain, batch_size=batch_size, epochs=epochs)

    def evaluate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        xTrain, yTrain = self._generate_data()
        yHat = self.model.predict(xTrain)
        return xTrain, yTrain, yHat

    @staticmethod
    def plot_results(xTrain: np.ndarray, yTrain: np.ndarray, yHat: np.ndarray) -> None:
        plt.plot(xTrain, yTrain, label='True', color='black')
        plt.plot(xTrain, yHat, label='Predicted', color='red')
        plt.legend()
        plt.show()
