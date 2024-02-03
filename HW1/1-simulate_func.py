import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class DNN1(nn.Module):
    def __init__(self):
        super(DNN1, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.main(x)

class DNN2(nn.Module):
    def __init__(self):
        super(DNN2, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(1, 20),
            nn.ReLU(),
            nn.Linear(20, 20, bias=False),
            nn.ReLU(),
            nn.Linear(20, 1),
        )

    def forward(self, x):
        return self.main(x)

class DNN3(nn.Module):
    def __init__(self):
        super(DNN3, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(1, 154),
            nn.ReLU(),
            nn.Linear(154, 1),
        )

    def forward(self, x):
        return self.main(x)


def train_model(model, train_loader, optimizer, criterion, epochs=1000):
    model.train()
    losses = []
    progress_bar = tqdm(range(epochs))
    for epoch in progress_bar:
        epoch_loss = 0
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        losses.append(epoch_loss)
        progress_bar.set_description(f'Epoch: {epoch+1}/{epochs}. Loss: {epoch_loss:.5f}')
    return losses

if __name__ == "__main__":
    # Create the model
    model1 = DNN1()
    model2 = DNN2()
    model3 = DNN3()

    # print the number of parameters in the model
    print(f'Number of parameters: DNN1={sum(p.numel() for p in model1.parameters())}, DNN2={sum(p.numel() for p in model2.parameters())}, DNN3={sum(p.numel() for p in DNN3().parameters())}')

    # Create the data
    x = torch.linspace(-10, 10, 10000).reshape(-1, 1)
    y = torch.sin(x) + torch.cos(2*x)

    # parameters
    train_loader = torch.utils.data.DataLoader(list(zip(x, y)), batch_size=2000, shuffle=True)
    lr = 1e-3
    optimizer1 = optim.Adam(model1.parameters(), lr=lr)
    optimizer2 = optim.Adam(model2.parameters(), lr=lr)
    optimizer3 = optim.Adam(model3.parameters(), lr=lr)
    criterion = nn.MSELoss()
    epochs = 1000

    # Train the model
    losses1 = train_model(model1, train_loader, optimizer1, criterion, epochs)
    losses2 = train_model(model2, train_loader, optimizer2, criterion, epochs)
    losses3 = train_model(model3, train_loader, optimizer3, criterion, epochs)

    # Plot the loss
    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(losses1, label='DNN1', color='cyan')
    plt.plot(losses2, label='DNN2', color='orange')
    plt.plot(losses3, label='DNN3', color='green')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.show()
    fig1.savefig('1-func_loss.png')
    plt.close()

    # Plot the prediction
    fig2 = plt.figure(figsize=(10, 5))
    plt.plot(x, y, label='GT', color='blue')
    plt.plot(x, model1(x).detach().numpy(), label='DNN1', color='cyan')
    plt.plot(x, model2(x).detach().numpy(), label='DNN2', color='orange')
    plt.plot(x, model3(x).detach().numpy(), label='DNN3', color='green')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.show()
    fig2.savefig('1-func_prediction.png')
    plt.close()

