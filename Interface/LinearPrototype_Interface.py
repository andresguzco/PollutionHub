import Engine.LinearPrototype as lp


def main() -> None:
    iN: int = int(1e5)

    sequenceModel: lp.SequenceModel = lp.SequenceModel(iN=iN)
    sequenceModel.train()
    sequenceModel.predict(plot=True)
    return None


if __name__ == '__main__':
    main()
