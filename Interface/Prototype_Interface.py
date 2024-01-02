from Engine.NNTemplate import GeneralNN


def main() -> None:
    Model = GeneralNN()
    Model.train()
    Model.predict(plot=True)
    return None


if __name__ == "__main__":
    main()
