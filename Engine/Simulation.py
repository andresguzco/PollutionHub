from Engine.Tools import phiFunction, relativeAngle, distanceCalc
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
from itertools import product
import matplotlib.cm as cm
import numpy as np
import random


class PollutionSimulation:
    def __init__(
            self,
            N: int = 1000,
            Lag: int = 1,
            Phi: float = 0.35,
            Rho: float = 0.5,
            timeInterval: int = 1,
            meanPol: float = 10.0,
            Distance: float = 10.0,
            rowNormalise: bool = False
    ):
        random.seed(123)
        self.N: int = N
        self.Lag: int = Lag
        self.Phi: float = Phi
        self.Rho: float = Rho
        self.Normalise = rowNormalise
        self.meanPol: float = meanPol
        self.Distance: float = Distance
        self.timeInterval: int = timeInterval
        self.GridSize: List[int] = [x for x in range(-1, 2)]
        self.Location: Dict[int, Tuple[int, int]] = {
            i: x for i, x in enumerate(list(product(self.GridSize, repeat=2)))
        }
        self.K: int = len(self.GridSize) ** 2
        self.windSpeed: np.ndarray = np.ones(shape=N) * 20
        self.windDirection: np.ndarray = np.random.uniform(low=0, high=360, size=[N])
        self.initialPollution: float = np.random.normal(loc=50, scale=2.5, size=[1, self.K])
        self.Y: np.ndarray = np.zeros([N, self.K])

    def compute_phi_values(self) -> np.ndarray:
        W = np.zeros((self.N, self.K, self.K))
        for k in range(self.N):
            for pair in product(list(self.Location.keys()), repeat=2):
                i, j = pair
                W[k, i, j] = 0 if i == j else phiFunction(
                    phi=self.Phi,
                    rho=self.Rho,
                    distance=distanceCalc(
                        coordinateA=self.Location[i],
                        coordinateB=self.Location[j],
                        unit=self.Distance
                    ),
                    n=self.timeInterval,
                    velocity=self.windSpeed[k],
                    angle=relativeAngle(
                        coordinateA=self.Location[i],
                        coordinateB=self.Location[j],
                        windDir=self.windDirection[k]
                    ),
                    lag=self.Lag
                )
            if self.Normalise:
                rowNorms: np.ndarray = np.linalg.norm(W[k, :, :], axis=1)
                rowNorms[rowNorms == 0] = 1
                W[k, :, :] = W[k, :, :] / rowNorms[:, np.newaxis]
        return W

    def simulate_pollution(self, W: np.ndarray) -> np.ndarray:
        self.Y[0, :] = self.meanPol + self.initialPollution
        for i in range(1, self.N):
            self.Y[i, :] = (
                    self.meanPol + W[i, :, :] @ self.Y[i - 1, :].transpose() +
                    np.random.normal(loc=0, scale=1, size=[1, self.K]))
        return self.Y

    @staticmethod
    def plot_results(Y: np.ndarray) -> None:
        fig, ax = plt.subplots()
        cmap = cm.get_cmap('viridis')
        colors = [cmap(i) for i in np.linspace(0, 1, Y.shape[1])]
        for i in range(Y.shape[1]):
            ax.plot(Y[:, i], color=colors[i])
        plt.show()


