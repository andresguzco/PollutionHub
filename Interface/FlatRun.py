from Engine.SARNet import CustomSARModel
import numpy as np


def main(N: int, K: int, F: int) -> None:
    X_test, X_train = np.zeros(shape=(N, N))
    Y_train, Y_test = np.ones(shape=(N, K))

    model = CustomSARModel(K, F)

    # Assume X_train, Y_train, X_test, Y_test are defined
    model.train(X_train, Y_train)
    loss, rho, W = model.evaluate(X_test, Y_test)
    model.plot_results(X_test, Y_test, rho, W)
    return None


if __name__ == "__main__":
    N = 1000
    K = 9
    F = 3
    main(N=N, K=K, F=F)
