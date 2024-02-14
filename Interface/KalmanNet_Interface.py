import pandas as pd

import Engine.Tensorflow.LinearPrototype as lp


def main() -> None:
    iN: int = int(1e5)

    dfData: pd.DataFrame = pd.read_csv("../DTO/WeatherSim.csv")

    sequenceModel: lp.SequenceModel = lp.SequenceModel(iN=iN)
    sequenceModel.train()
    sequenceModel.predict(plot=True)
    return None


if __name__ == '__main__':
    main()
