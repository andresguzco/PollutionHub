import numpy as np
import math


def relativeAngle(
        coordinateA: list,
        coordinateB: list = None,
        windDir: float = 0.0
) -> float:
    unit_x_axis = np.array([1, 0])
    if np.linalg.norm(np.array(coordinateA)) == 0:
        v1_u: np.ndarray = unit_x_axis
    else:
        v1_u: np.ndarray = np.array(coordinateA) / np.linalg.norm(np.array(coordinateA))
    if np.linalg.norm(np.array(coordinateB)) == 0:
        v2_u: np.ndarray = unit_x_axis
    else:
        v2_u: np.ndarray = np.array(coordinateB) / np.linalg.norm(np.array(coordinateB))
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) - windDir


def phiFunction(
        phi: float,
        rho: float,
        distance: float,
        n: int,
        velocity: float,
        angle: float,
        lag: int
) -> float:
    result: float = phi - rho * abs(((distance * n)/(velocity * np.cos(angle))) - lag)
    return result if result > 0 else 0.0


def distanceCalc(coordinateA: list, coordinateB: list, unit: float) -> float:
    return math.dist(coordinateA, coordinateB) * unit


def estimateModel() -> object:
    return object
