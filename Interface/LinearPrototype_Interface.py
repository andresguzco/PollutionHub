import Engine.LinearPrototype as lp


def main() -> None:
    iN: int = int(1e5)
    x_train: lp.np.ndarray
    y_train: lp.np.ndarray
    yy_train: lp.np.ndarray
    x_train, y_train, yy_train = lp.prepareData(iN)

    sequence_model: lp.SequenceModel = lp.SequenceModel()
    sequence_model.train(x_train, yy_train, iN)

    y_hat: lp.np.ndarray[lp.Any, lp.np.dtype[lp.Any]] = sequence_model.predict(x_train, iN)
    print(sequence_model.fnLoss(yy_train, y_hat))
    lp.plotPrediction(x_train, y_train, y_hat)
    return None


if __name__ == '__main__':
    main()
