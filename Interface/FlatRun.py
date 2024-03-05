from Engine.Pipeline import Pipeline
from Engine.STFNet import STFNet
import torch.optim as optim
import torch.nn as nn


class FlatRun:
    def __init__(self):
        self.Model = STFNet()
        self.Loss = nn.MSELoss()
        self.Optimizer = optim.Adam(self.Model.parameters(), lr=0.001)

    def Run(self, trainData, valData, testData):
        pipeline = Pipeline(
            trainDataset=trainData,
            valDataset=valData,
            testDataset=testData,
            model=self.Model,
            optimizer=self.Optimizer,
            lossFunction=self.Loss,
            batchSize=64,
            useCuda=True
        )
        pipeline.run(epochs=100)
        return


def main() -> None:
    trainDataset = ...  # Your training dataset
    valDataset = ...    # Your validation dataset
    testDataset = ...   # Your test dataset

    Process = FlatRun()
    Process.Run(trainDataset, valDataset, testDataset)
    return None


if __name__ == '__main__':
    main()
