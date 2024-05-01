from torch.utils.data import DataLoader, TensorDataset
from Engine.STFNet import STFNet
from torch import optim
import pandas as pd
import torch
import math


def CustomLoss(y, mu, variances):
    k = mu.size(1)
    nll_total = 0
    batch_size = mu.size(0)

    for i in range(1, batch_size):
        term1 = torch.logdet(variances[i]) + k * torch.log(torch.tensor(2 * math.pi))
        diff = y[i] - mu[i]
        term2 = torch.sum(diff @ torch.inverse(variances[i]) * diff)
        nll = 0.5 * (term1 + term2)
        nll_total += nll

    mean_nll = nll_total / (batch_size - 1)
    return mean_nll


class Pipeline:
    def __init__(self, dim_y: int = 106, output_dims: int = 7):
        self.model = STFNet(dim_y, output_dims)

    def run(self, train_loader, epochs=100):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        torch.autograd.set_detect_anomaly(True)

        for epoch in range(epochs):
            total_loss = 0
            for i, (X, Y) in enumerate(train_loader):
                y_pred, var = self.model.forward(Y, X)
                try:
                    loss = CustomLoss(Y, y_pred, var)
                    total_loss += loss.item()
                except Exception as e:
                    print(f'Error in Loss. Epoch {epoch}. Batch {i}. Error: {str(e)}')
                    break

                optimizer.zero_grad()
                loss.backward()
                try:
                    loss.backward()
                except Exception as e:
                    print(f'Error during backpropagation. Epoch {epoch}. Batch {i}. Error: {str(e)}')
                continue

                optimizer.step()
                if i % 10 == 0:
                    print(f'Epoch {epoch + 1}/{epochs}, Batch {i}, Loss: {loss.item()}')

        print('Training complete.')


def main() -> None:
    mainDataset = pd.read_csv("../DTO/CleanData.csv")
    mainDataset.drop('timestamp', axis=1, inplace=True)

    YDf = torch.Tensor(mainDataset.iloc[1:, :7].values)
    ZDf = torch.Tensor(mainDataset.diff().iloc[1:, :].values)
    trainData = DataLoader(TensorDataset(ZDf, YDf), batch_size=1)

    trainer = Pipeline()
    trainer.run(train_loader=trainData)
    return None


if __name__ == '__main__':
    main()
