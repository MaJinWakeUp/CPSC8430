import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
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

def model_gradients_norm(model, p=2):
    gradients = []
    for param in model.parameters():
        gradients.append(param.grad.flatten())
    gradients = torch.cat(gradients)
    return torch.norm(gradients, p)

def train_model(model, train_loader, optimizer, criterion, epochs=1000):
    model.train()
    losses = []
    gradients = []
    progress_bar = tqdm(range(epochs))
    for epoch in progress_bar:
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            gradients.append(model_gradients_norm(model).item())
        progress_bar.set_description(f'Epoch: {epoch+1}/{epochs}')
    return losses, gradients

if __name__ == "__main__":
    # Create the model
    model = DNN().to(device)

    # Create the data
    x = torch.linspace(-10, 10, 10000).reshape(-1, 1)
    y = torch.sin(2 * torch.pi * x) / (2 * torch.pi * x)

    # parameters
    train_loader = torch.utils.data.DataLoader(list(zip(x, y)), batch_size=2000, shuffle=True)
    lr = 1e-3
    optimizer1 = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    epochs = 1000

    # train the model
    losses, gradients = train_model(model, train_loader, optimizer1, criterion, epochs)

    # plot the loss and gradients
    figure = plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.plot(losses)
    plt.title('Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.subplot(2, 1, 2)
    plt.plot(gradients)
    plt.title('Gradients')
    plt.xlabel('Iterations')
    plt.ylabel('Gradients')
    plt.show()
    figure.savefig('./figs/2-func_gradients.png')

