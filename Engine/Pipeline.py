from torch.utils.data import DataLoader, Dataset
import torch


class Pipeline:
    def __init__(self, trainDataset: Dataset, valDataset: Dataset, testDataset: Dataset,
                 model: torch.nn.Module, optimizer: torch.optim.Optimizer, lossFunction: torch.nn.Module,
                 batchSize: int = 64, useCuda: bool = True):
        self.trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
        self.valLoader = DataLoader(valDataset, batch_size=batchSize)
        self.testLoader = DataLoader(testDataset, batch_size=batchSize)
        self.model = model.to('cuda' if useCuda and torch.cuda.is_available() else 'cpu')
        self.optimizer = optimizer
        self.lossFunction = lossFunction
        self.device = 'cuda' if useCuda and torch.cuda.is_available() else 'cpu'

    def trainEpoch(self) -> float:
        self.model.train()
        totalLoss = 0.0
        for X_batch, y_batch in self.trainLoader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            self.optimizer.zero_grad()
            predictions = self.model(X_batch)
            loss = self.lossFunction(predictions, y_batch)
            loss.backward()
            self.optimizer.step()
            totalLoss += loss.item()
        avgLoss = totalLoss / len(self.trainLoader)
        return avgLoss

    def validate(self) -> float:
        self.model.eval()
        totalLoss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in self.valLoader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                predictions = self.model(X_batch)
                loss = self.lossFunction(predictions, y_batch)
                totalLoss += loss.item()
        avgLoss = totalLoss / len(self.valLoader)
        return avgLoss

    def evaluate(self) -> float:
        self.model.eval()
        totalLoss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in self.testLoader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                predictions = self.model(X_batch)
                loss = self.lossFunction(predictions, y_batch)
                totalLoss += loss.item()
        avgLoss = totalLoss / len(self.testLoader)
        return avgLoss

    def run(self, epochs: int = 100) -> None:
        for epoch in range(epochs):
            trainLoss = self.trainEpoch()
            valLoss = self.validate()
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {trainLoss:.4f}, Validation Loss: {valLoss:.4f}')

        testLoss = self.evaluate()
        print(f'Test Loss: {testLoss:.4f}')
