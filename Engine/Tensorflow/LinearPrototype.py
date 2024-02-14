from tensorflow import reduce_sum, Tensor
from Engine.Tensorflow.NNTemplate import GeneralNN
from numpy import ndarray


def fnLoss(y: ndarray, yPred: ndarray) -> Tensor:
    e: ndarray = y[:, 0] - y[:, 1] * yPred[:, 0]
    return reduce_sum(e * e)


class SequenceModel(GeneralNN):
    def __init__(self, iN: int):
        super().__init__(iN=iN, Loss=fnLoss)
