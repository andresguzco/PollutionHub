import matplotlib.pyplot as plt
from typing import Tuple
import tensorflow as tf
import numpy as np


class CustomModel:
    def __init__(self, input_size: int = 100000) -> None:
        self.input_size: int = input_size
        self.model: tf.keras.Sequential = self._build_model()

    def _generate_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        xTrain = np.sort(np.random.uniform(size=self.input_size) - 0.5).reshape(self.input_size, 1)
        yTrain = 2 + np.cos(4 * np.pi * xTrain)
        XX = np.random.normal(5, size=self.input_size).reshape(self.input_size, 1)
        YY = XX * yTrain + np.random.normal(0, 0.01, size=self.input_size).reshape(self.input_size, 1)
        yyTrain = np.concatenate([YY, XX], axis=1)
        return xTrain, yTrain, yyTrain

    @staticmethod
    def _build_model() -> tf.keras.Sequential:
        Model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(1, 1)),
            tf.keras.layers.Dense(6, activation='sigmoid'),
            tf.keras.layers.Dense(3, activation='sigmoid'),
            tf.keras.layers.Dense(2)
        ])
        return Model

    @staticmethod
    def _custom_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        e = y_true[:, 0] - y_true[:, 1] * y_pred[:, 0]
        return tf.reduce_sum(e * e)

    def train(self, epochs: int = 250, batch_size: int = 10000) -> None:
        xTrain, _, yyTrain = self._generate_data()
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=self._custom_loss)
        self.model.fit(xTrain, yyTrain, batch_size=batch_size, epochs=epochs)

    def evaluate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        xTrain, yTrain, yyTrain = self._generate_data()
        self.model.evaluate(xTrain, yyTrain, batch_size=self.input_size, verbose=2)
        yHat = self.model.predict(xTrain)
        print("Custom Loss:", self._custom_loss(yyTrain, yHat).numpy())
        return xTrain, yTrain, yHat

    @staticmethod
    def plot_results(xTrain: np.ndarray, yTrain: np.ndarray, yHat: np.ndarray) -> None:
        yHat = yHat[:, 0]
        yMin = min(np.min(yHat), np.min(yTrain))
        yMax = max(np.max(yHat), np.max(yTrain))
        plt.plot(xTrain, yTrain, label='True', color='black')
        plt.plot(xTrain, yHat, label='Predicted', color='red')
        plt.ylim([yMin, yMax])
        plt.legend()
        plt.show()
