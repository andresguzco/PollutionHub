import matplotlib.pyplot as plt
from numpy import ndarray, dtype
from typing import Any, Callable
import tensorflow as tf
from typing import Dict
import numpy as np


class GeneralNN:
    def __init__(
            self,
            iN: int = 1000,
            ModelConfig: Dict[str, Any] = None,
            Loss: Callable = None,
            Data: Dict[str, Any] = None,
            Optimiser: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    ):
        if not ModelConfig:
            ModelConfig = {
                'layers': [
                    {'type': 'Dense', 'params': {'units': 64, 'activation': 'relu', 'input_shape': (1,)}},
                    {'type': 'Dense', 'params': {'units': 64, 'activation': 'relu'}},
                    {'type': 'Dense', 'params': {'units': 1}}
                ]
            }
        self.N = iN
        self.data: Dict[str, np.ndarray] = Data if Data else self.__prepareData__()
        self.model: tf.keras.Sequential = self.__createModel__(ModelConfig)
        self.model.compile(optimizer=Optimiser, loss=Loss if Loss else 'mean_squared_error')

    @staticmethod
    def __createModel__(config):
        model = tf.keras.Sequential()
        for layer_config in config['layers']:
            layer_type = layer_config['type']
            if layer_type == 'Dense':
                model.add(tf.keras.layers.Dense(**layer_config['params']))
            elif layer_type == 'SimpleRNN':
                model.add(tf.keras.layers.SimpleRNN(**layer_config['params']))
            elif layer_type == 'LSTM':
                model.add(tf.keras.layers.LSTM(**layer_config['params']))
            elif layer_type == 'GRU':
                model.add(tf.keras.layers.GRU(**layer_config['params']))
        return model

    def predict(self, plot: bool) -> ndarray[Any, dtype[Any]]:
        yHat = self.model.predict(self["xTrain"])
        if plot:
            self.__plotPrediction__(yHat=yHat)
        return yHat

    def train(self) -> None:
        self.model.fit(self["xTrain"], self["yTrain"], batch_size=250, epochs=250)
        self.model.evaluate(self["xTrain"], self["yTrain"], batch_size=self.N, verbose=2)
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
        return {"xTrain": xTrain, "yTrain": yyTrain}

    def __getitem__(self, item):
        return self.data[item]
