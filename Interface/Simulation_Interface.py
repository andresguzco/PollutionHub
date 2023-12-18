from Engine.Simulation import PollutionSimulation
from numpy import ndarray, linspace
from tqdm import tqdm
import pandas as pd
import numpy as np


def gridSearch(

        phi_min: float,
        phi_max: float,
        rho_min: float,
        rho_max: float,
        num_steps: int,
        save: bool = False,
        N: int = 1000
):
    phi_values: ndarray = linspace(phi_min, phi_max, num_steps)
    rho_values: ndarray = linspace(rho_min, rho_max, num_steps)

    for phi in phi_values:
        for rho in tqdm(rho_values):
            try:
                simulation: PollutionSimulation = PollutionSimulation(
                    N=N,
                    Lag=1,
                    Phi=phi,
                    Rho=rho,
                    timeInterval=1,
                    meanPol=10.0,
                    Distance=10.0,
                    muWind=20
                )
                W: ndarray = simulation.computePhi()
                print(W[20, :, :])
                Y: ndarray = simulation.simulateVar(W)

                filename: str = f"../Output/phi_{phi:.2f}_rho_{rho:.2f}.png"
                simulation.plot_results(Y, filename)

                Database: np.ndarray = np.zeros(shape=(N - 100, 2))
                Database[:, 0] = simulation.windSpeed
                Database[:, 1] = simulation.windDirection

                if save:
                    np.save("../DTO/SpatialTensor.npy", W)
                    pd.DataFrame(Y).to_csv("../DTO/PollutionSim.csv", header=False, index=False)
                    pd.DataFrame(Database).to_csv("../DTO/WeatherSim.csv", header=False, index=False)
                    pd.DataFrame(simulation.Location).to_csv("../DTO/LocationGrid.csv",
                                                             header=False, index=False)
            except OverflowError:
                print(f"Overflow error for Phi = {phi:.2f}, Rho = {rho:.2f}. Skipping.")
                continue
    return None


def main() -> None:
    gridSearch(
        N=1100,
        phi_min=0.40,
        phi_max=0.40,
        rho_min=0.40,
        rho_max=0.40,
        num_steps=1,
        save=True
    )
    return None


if __name__ == "__main__":
    main()
