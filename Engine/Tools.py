import numpy as np


def relativeAngle(coordinateA: list, coordinateB: list, windDir: float) -> float:
    v1_u = np.array(coordinateA) / np.linalg.norm(np.array(coordinateA))
    v2_u = np.array(coordinateB) / np.linalg.norm(np.array(coordinateB))
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) - windDir


def phiFunction(phi: float, rho: float, distance: float, n: int, velocity: float, angle: float, lag: int) -> float:
    return abs(phi - rho * (((distance * n)/(velocity * np.cos(angle))) - lag)**2)
