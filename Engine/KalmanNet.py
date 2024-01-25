from Engine.NNTemplate import GeneralNN
from typing import Dict
import tensorflow as tf
import numpy as np


# TODO: Add structure for the covariates of the time-series
# TODO: Discuss the spatial structure of the data.
#  Should it include coordinates? Should it only include weather covariates?


class MultivariateKalmanFilter:
    def __init__(
            self,
            A: np.ndarray = np.eye(9),
            H: np.ndarray = np.eye(9),
            Q: np.ndarray = np.eye(9),
            R: np.ndarray = np.eye(9),
            x0: np.ndarray = np.zeros(9),
            P0: np.ndarray = np.eye(9)
    ):
        self.A: np.ndarray = A   # State transition matrix
        self.H: np.ndarray = H   # Observation matrix
        self.Q: np.ndarray = Q   # Process noise covariance
        self.R: np.ndarray = R   # Measurement noise covariance
        self.x: np.ndarray = x0  # Initial state
        self.P: np.ndarray = P0  # Initial covariance estimate

    def predict(self) -> np.ndarray:
        self.x = np.dot(self.A, self.x)
        self.P = np.dot(self.A, np.dot(self.P, self.A.T)) + self.Q
        return self.x

    def update(self, measurement) -> None:
        y: np.ndarray = measurement - np.dot(self.H, self.x)
        S: np.ndarray = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K: np.ndarray = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(S)))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(K, np.dot(self.H, self.P))
        return None


def kalmanLoss(y_true, y_pred, kalman_filter):
    kalman_filter.update(y_true)
    kf_estimate = kalman_filter.predict()
    return tf.reduce_mean(tf.square(y_pred - kf_estimate))


class KalmanNet(GeneralNN):
    def __init__(
            self,
            iN: int,
            Data: Dict[str, np.ndarray] = None,
            timeSteps: int = None,
            nFeatures: int = None,
            nOutput: int = None
    ) -> None:
        ModelConfig: Dict[str, list] = {
            'layers': [
                # LSTM layer for capturing temporal dynamics
                {'type': 'LSTM', 'params': {
                    'units': 50, 'activation': 'relu', 'input_shape': (timeSteps, nFeatures), 'return_sequences': True}
                 },
                # Additional LSTM Layer
                {'type': 'LSTM', 'params': {
                    'units': 50, 'activation': 'relu', 'return_sequences': False}
                 },
                # Dropout for regularization
                {'type': 'Dropout', 'params': {'rate': 0.2}},
                # Batch Normalization
                {'type': 'BatchNormalization'},
                # Dense layer for output
                {'type': 'Dense', 'params': {'units': nOutput, 'activation': 'linear'}}
            ]
        }

        super().__init__(
            iN=iN,
            Data=Data,
            ModelConfig=ModelConfig,
            Loss=lambda y_true, y_pred: kalmanLoss(y_true, y_pred, MultivariateKalmanFilter())
        )
