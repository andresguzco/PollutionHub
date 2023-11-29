from Engine.Simulation import PollutionSimulation


def main() -> None:
    simulation = PollutionSimulation()
    W = simulation.compute_phi_values()
    Y = simulation.simulate_pollution(W)
    simulation.plot_results(Y)
    return None


if __name__ == "__main__":
    main()
