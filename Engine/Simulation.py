import matplotlib.pyplot as plt
from itertools import product
import matplotlib.cm as cm
from typing import List
from numba import jit, float64
import numpy as np
import random


# TODO: Label locations
# TODO: Initialise to the conditional mean
# TODO: Separate heatmaps per column to observe dynamics in each locations. Tessellate the heatmap
# TODO: Visualisation for a finely grain mesh and the funnel of the effects of the pollution dispersion


class PollutionSimulation(object):

    def __init__(
            self,
            N: int = 1000,
            Lag: int = 1,
            Phi: float = 0.35,
            Rho: float = 0.50,
            timeInterval: int = 1,
            meanPol: float = 10.0,
            Distance: float = 10.0,
            muWind: float = 20.0,
            phiWind: float = 0.8,
            fixWindSpeed: bool = True,
            fixWindDirection: bool = True,
            formulation: str = "Quadratic"
    ):
        random.seed(123)

        self.N: int = N
        self.Lag: int = Lag
        self.Phi: float = Phi
        self.Rho: float = Rho

        self.meanPol: float = meanPol
        self.Distance: float = Distance
        self.timeInterval: int = timeInterval
        self.phiWind: float = phiWind

        self.GridSize: List[int] = [x for x in range(-1, 2)]
        self.Location: np.ndarray = np.array(list(product(self.GridSize, repeat=2)))
        self.K: int = len(self.GridSize) ** 2

        if fixWindSpeed:
            self.windSpeed: np.ndarray = np.random.normal(muWind, 1, N)
        else:
            self.windSpeed: np.ndarray = np.ones(N) * muWind
        if fixWindDirection:
            self.windDirection: np.ndarray = np.ones(N) * 180
        else:
            self.windDirection: np.ndarray = np.random.uniform(low=0, high=360, size=[N])

        self.initialPollution: float = np.random.normal(loc=50, scale=2.5, size=[1, self.K])
        self.Y: np.ndarray = np.zeros([N, self.K])
        self.formulation = formulation

    # def updateWindSpeed(self) -> None:
    #     epsilon = np.random.normal(0, 1, self.N)
    #     self.windSpeed = self.muWind + self.phiWind * (self.windSpeed - self.muWind) + epsilon
    #     return None

    def computePhi(self) -> np.ndarray:
        return self._phiComputation(
            Phi=self.Phi,
            Rho=self.Rho,
            timeInterval=self.timeInterval,
            windSpeed=self.windSpeed,
            windDirection=self.windDirection,
            Lag=self.Lag,
            N=self.N,
            K=self.K,
            Location=self.Location,
            Distance=self.Distance,
            formulation=self.formulation
        )

    @staticmethod
    @jit(nopython=True)
    def _phiComputation(
            Phi: float,
            Rho: float,
            timeInterval,
            windSpeed: np.ndarray,
            windDirection: np.ndarray,
            Lag: int,
            N: int,
            K: int,
            Location: np.ndarray,
            Distance: float,
            formulation: str = "Quadratic"
    ) -> np.ndarray:
        W: np.ndarray = np.zeros((N, K, K))
        for k in range(N):
            for i in range(K):
                for j in range(K):
                    W[k, i, j] = 0 if i == j else phiFunction(
                        phi=Phi,
                        rho=Rho,
                        distance=distanceCalc(
                            coordinateA=Location[i],
                            coordinateB=Location[j],
                            unit=Distance
                        ),
                        n=timeInterval,
                        velocity=windSpeed[k],
                        angle=relativeAngle(
                            coordinateA=Location[i],
                            coordinateB=Location[j],
                            windDir=windDirection[k]
                        ),
                        lag=Lag,
                        formulation=formulation
                    )
        return W

    def simulateVar(self, W_input: np.ndarray) -> np.ndarray:
        Yinput: np.ndarray = np.zeros([self.N, self.K])
        self.windSpeed = self.windSpeed[100:]
        self.windDirection = self.windDirection[100:]
        return self._simComputation(
            N=self.N,
            K=self.K,
            meanPol=self.meanPol,
            initialPollution=self.initialPollution,
            W=W_input,
            Y=Yinput
        )

    @staticmethod
    @jit(nopython=True)
    def _simComputation(
            N: int,
            K: int,
            meanPol: float,
            initialPollution: float,
            W: np.ndarray,
            Y: np.ndarray
    ) -> np.ndarray:
        Y[0, :] = meanPol + initialPollution
        for i in range(1, N):
            Y[i, :] = (
                    meanPol + np.dot(W[i, :, :], Y[i - 1, :].transpose()) +
                    np.random.normal(loc=0, scale=1, size=(1, K)))
        return Y[100:, :]

    @staticmethod
    def plot_results(Y: np.ndarray, filepath: str) -> None:
        if not np.all(np.isfinite(Y)) or np.all(Y == Y[0]):
            print(f"Invalid data in Y, skipping plot for {filepath}")
            return None

        fig, ax = plt.subplots()
        cmap = cm.get_cmap('viridis')
        colors = [cmap(i) for i in np.linspace(0, 1, Y.shape[1])]
        for i in range(Y.shape[1]):
            ax.plot(Y[:, i], color=colors[i])
        plt.savefig(filepath)
        plt.clf()
        return None


@jit(nopython=True)
def relativeAngle(coordinateA, coordinateB, windDir):
    coordinateA = coordinateA.astype(float64)
    coordinateB = coordinateB.astype(float64)
    v1_u = coordinateA / np.linalg.norm(coordinateA)
    v2_u = coordinateB / np.linalg.norm(coordinateB)
    dot_product = np.dot(v1_u, v2_u)
    clipped_dot_product = max(min(dot_product, 1.0), -1.0)
    return np.arccos(clipped_dot_product) - windDir


@jit(nopython=True)
def phiFunction(
        phi: float,
        rho: float,
        distance: float,
        n: int,
        velocity: np.ndarray,
        angle: float,
        lag: int,
        formulation: str = "Absolute"
) -> float:
    if formulation == "Absolute":
        result = phi - rho * abs(((distance * n) / (velocity * np.cos(angle))) - lag)
    else:
        result = phi - rho * (((distance * n) / (velocity * np.cos(angle))) - lag) ** 2
    return result if result > 0 else 0.0


@jit(nopython=True)
def distanceCalc(coordinateA, coordinateB, unit):
    coordinateA = coordinateA.astype(float64)
    coordinateB = coordinateB.astype(float64)
    diff = coordinateA - coordinateB
    return np.sqrt(np.dot(diff, diff)) * unit
