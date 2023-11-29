from Engine.Prototype import CustomModel


def main() -> None:
    Model = CustomModel()
    Model.train()
    xTrain, yTrain, yHat = Model.evaluate()
    Model.plot_results(xTrain, yTrain, yHat)
    return None


if __name__ == "__main__":
    main()
