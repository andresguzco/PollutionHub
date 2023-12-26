from tensorflow.keras import layers
from numpy import ndarray, dtype
import matplotlib.pyplot as plt
from typing import Tuple, Any
from tensorflow import keras
import tensorflow as tf
import numpy as np


class SequenceModel(object):
    def __init__(self):
        self.model: keras.Sequential = keras.Sequential([
            layers.Flatten(input_shape=(1, 1)),
            layers.Dense(6, activation="sigmoid"),
            layers.Dense(3, activation="sigmoid"),
            layers.Dense(2)
        ])
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=self.fnLoss
        )

    @staticmethod
    def fnLoss(y: np.ndarray, yPred: np.ndarray) -> tf.Tensor:
        e: np.ndarray = y[:, 0] - y[:, 1] * yPred[:, 0]
        return tf.reduce_sum(e * e)

    def train(self, xTrain: np.ndarray, yyTrain: np.ndarray, iN: int) -> None:
        self.model.fit(xTrain, yyTrain, batch_size=int(1e4), epochs=250)
        self.model.evaluate(xTrain, yyTrain, batch_size=iN, verbose=2)
        return None

    def predict(self, x_train: np.ndarray, iN: int) -> ndarray[Any, dtype[Any]]:
        return np.reshape(self.model.predict(x_train), (iN, 2))


def prepareData(iN: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xTrain: np.ndarray = np.sort(np.random.uniform(size=iN)) - 0.5
    yTrain: np.ndarray = 2 + np.cos(4 * np.pi * xTrain)
    xx: np.ndarray = np.random.normal(loc=5, size=iN)
    yy: np.ndarray = xx * yTrain + np.random.normal(scale=0.01, size=iN)
    yyTrain: np.ndarray = np.c_[yy, xx]
    return xTrain, yTrain, yyTrain


def plotPrediction(xTrain: np.ndarray, yTrain: np.ndarray, yHat: np.ndarray) -> None:
    Aid = np.concatenate([yHat, yTrain])
    yMin: float = np.min(Aid)
    yMax: float = np.max(Aid)
    plt.plot(xTrain, yTrain, 'k')
    plt.ylim(yMin, yMax)
    plt.plot(xTrain, yHat, 'r')
    plt.show()
    return None
