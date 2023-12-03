from Engine.Simulation import PollutionSimulation
from numpy import ndarray, linspace
from tqdm import tqdm


def gridSearch(
        phi_min: float,
        phi_max: float,
        rho_min: float,
        rho_max: float,
        num_steps: int
):
    phi_values: ndarray = linspace(phi_min, phi_max, num_steps)
    rho_values: ndarray = linspace(rho_min, rho_max, num_steps)

    for phi in phi_values:
        for rho in tqdm(rho_values):
            try:
                simulation: PollutionSimulation = PollutionSimulation(
                    N=1000,
                    Lag=1,
                    Phi=phi,
                    Rho=rho,
                    timeInterval=1,
                    meanPol=10.0,
                    Distance=10.0,
                    muWind=20,
                    phiWind=0.7
                )
                W: ndarray = simulation.computePhi()
                Y: ndarray = simulation.simulateVar(W)

                filename: str = f"../Output/phi_{phi:.2f}_rho_{rho:.2f}.png"
                simulation.plot_results(Y, filename)
            except OverflowError:
                print(f"Overflow error for Phi = {phi:.2f}, Rho = {rho:.2f}. Skipping.")
                continue
    return None


def main() -> None:
    gridSearch(
        phi_min=0.22,
        phi_max=0.22,
        rho_min=0.22,
        rho_max=0.22,
        num_steps=1
    )
    return None


if __name__ == "__main__":
    main()
