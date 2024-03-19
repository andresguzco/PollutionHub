from torch.utils.data import DataLoader, TensorDataset

from Engine.STFNet import STFNet
from torch import optim
import torch.nn as nn
import torch
import pandas as pd


class Pipeline:
    def __init__(self, time_steps, dim_y: int = 106, output_dims: int = 7):
        self.model = STFNet(dim_y, time_steps, output_dims)

    def run(self, train_loader, epochs=100):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(epochs):

            for X, Y in train_loader:
                y_pred = self.model.forward(Y, X)
                loss = loss_fn(y_pred, Y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

        print('Training complete.')


def main() -> None:
    mainDataset = pd.read_csv("../DTO/CleanData.csv")
    mainDataset.drop('timestamp', axis=1, inplace=True)

    YDf = torch.Tensor(mainDataset.iloc[1:, :7].values)
    ZDf = torch.Tensor(mainDataset.diff().iloc[1:, :].values)
    trainData = DataLoader(TensorDataset(ZDf, YDf), batch_size=50)

    trainer = Pipeline(time_steps=len(ZDf))
    trainer.run(train_loader=trainData)
    return None


if __name__ == '__main__':
    main()
