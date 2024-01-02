from Engine.NNTemplate import GeneralNN


def main() -> None:
    iN: int = int(1e5)

    generalModel = GeneralNN(iN=iN)
    generalModel.train()
    generalModel.predict(plot=True)
    return None


if __name__ == "__main__":
    main()
