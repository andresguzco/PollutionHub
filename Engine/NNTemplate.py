import matplotlib.pyplot as plt
from numpy import ndarray, dtype
from typing import Any, Callable
import tensorflow as tf
from typing import Dict
import numpy as np


class GeneralNN:
    def __init__(
            self,
            iN: int,
            Loss: Callable = None,
            Data: Dict[str, Any] = None,
            Model: tf.keras.Model = None,
            Optimiser: tf.keras.optimizers.Optimiser = None
    ):
        self.N: int = iN
        self.data: Dict[str, np.ndarray] = Data if Data else self.__prepareData__()
        self.model: tf.keras.Sequential = Model if Model else tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        self.model.compile(
            optimizer=Optimiser if Optimiser else tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=Loss if Loss else 'mean_squared_error'
        )

    def predict(self, plot: bool) -> ndarray[Any, dtype[Any]]:
        yHat = self.model.predict(self["xTrain"])
        if plot:
            self.__plotPrediction__(yHat=yHat)
        return yHat

    def train(self) -> None:
        self.model.fit(self["xTrain"], self["yyTrain"], batch_size=250, epochs=250)
        self.model.evaluate(self["xTrain"], self["yyTrain"], batch_size=self.N, verbose=2)
        return None

    def __plotPrediction__(self, yHat) -> None:
        plt.plot(self["xTrain"], self["yTrain"], label='True', color='black')
        plt.plot(self["xTrain"], yHat, label='Predicted', color='red')
        plt.legend()
        plt.show()
        return None

    def __prepareData__(self) -> Dict[str, np.ndarray]:
        xTrain: np.ndarray = np.sort(np.random.uniform(size=self.N)) - 0.5
        yTrain: np.ndarray = 2 + np.cos(4 * np.pi * xTrain)
        xx: np.ndarray = np.random.normal(loc=5, size=self.N)
        yy: np.ndarray = xx * yTrain + np.random.normal(scale=0.01, size=self.N)
        yyTrain: np.ndarray = np.c_[yy, xx]
        return {"xTrain": xTrain, "yTrain": yTrain, "yyTrain": yyTrain}

    def __getitem__(self, item):
        return self.data[item]
