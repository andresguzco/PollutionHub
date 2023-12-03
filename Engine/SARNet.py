import matplotlib.pyplot as plt
from typing import Tuple
import tensorflow as tf
import numpy as np


class SARNetworkModel:
    def __init__(self, numLocations: int = 9, inputDim: int = 1) -> None:
        self.num_locations: int = numLocations
        self.input_dim: int = inputDim  # Dimension of X
        self.model: tf.keras.Model = self._build_model()

    def _build_model(self) -> tf.keras.Model:
        # Input for X variable
        input_X = tf.keras.Input(shape=(self.input_dim,))

        # Branch to learn rho
        rho_branch = tf.keras.layers.Dense(64, activation='relu')(input_X)
        rho = tf.keras.layers.Dense(1, activation='tanh')(rho_branch)  # tanh to bound rho between -1 and 1

        # Branch to learn W representation
        W_branch = tf.keras.layers.Dense(64, activation='relu')(input_X)
        W = tf.keras.layers.Dense(self.num_locations * self.num_locations, activation='sigmoid')(W_branch)
        W = tf.keras.layers.Reshape((self.num_locations, self.num_locations))(W)  # Reshape to matrix

        model = tf.keras.Model(inputs=input_X, outputs=[rho, W])
        return model

    def train(self, X: np.ndarray, Y: np.ndarray, epochs: int = 100, batch_size: int = 32) -> None:
        self.model.compile(optimizer='adam', loss=self._custom_loss)
        self.model.fit(X, Y, batch_size=batch_size, epochs=epochs)

    @staticmethod
    def _custom_loss(y_true: tf.Tensor, y_pred: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        # Custom loss function considering the SAR model structure
        rho, W = y_pred
        y_pred = rho * tf.matmul(W, y_true)  # Simplified SAR equation, matrix multiplication
        return tf.reduce_mean(tf.square(y_true - y_pred))

    def evaluate(self, X: np.ndarray, Y: np.ndarray) -> Tuple[tf.Tensor, np.ndarray, np.ndarray]:
        # Evaluate the model
        loss = self.model.evaluate(X, Y, verbose=0)
        y_pred = self.model.predict(X)

        # Extract rho and W for further analysis or plotting
        rho, W = self.model(X)
        return loss, rho.numpy(), W.numpy()

    @staticmethod
    def plot_results(X: np.ndarray, Y: np.ndarray, rho: np.ndarray, W: np.ndarray) -> None:
        # Assuming the first dimension of X is related to the spatial locations
        plt.figure(figsize=(12, 6))

        # Plotting rho values
        plt.subplot(1, 2, 1)
        plt.title("Estimated Rho Values")
        plt.plot(X[:, 0], rho, 'o')
        plt.xlabel("X[0]")
        plt.ylabel("Rho")

        # Plotting W matrix as a heatmap
        plt.subplot(1, 2, 2)
        plt.title("Estimated W Matrix")
        plt.imshow(W, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.xlabel("Location")
        plt.ylabel("Location")

        plt.tight_layout()
        plt.show()
