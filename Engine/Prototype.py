from Engine.NNTemplate import GeneralNN


class CustomModel(GeneralNN):
    def __init__(self, inputSize: int = 100000) -> None:
        super().__init__(iN=inputSize)

