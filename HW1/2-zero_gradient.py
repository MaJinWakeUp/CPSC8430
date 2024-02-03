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

def model_gradients(model):
    gradients = []
    for param in model.parameters():
        if param.grad is not None:
            gradients.append(param.grad.flatten())
    gradients = torch.cat(gradients) if gradients else torch.tensor(0.0)
    return gradients

def compute_minimal_ratio(gradients):
    positive_gradients = gradients[gradients > 0]
    minimal_ratio = positive_gradients.size(0) / gradients.size(0)
    return minimal_ratio

def train_model_to_zero_gradient(model, train_loader, optimizer, criterion, epochs=1000):
    model.train()
    losses = []
    minimal_ratios = []
    progress_bar = tqdm(range(epochs))
    for epoch in progress_bar:
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            
            if epoch < 500:
                loss = criterion(y_hat, y)
                loss.backward()
                optimizer.step()
            else:
                gradient = model_gradients(model)
                gradient_norm = torch.linalg.vector_norm(gradient)
                loss = criterion(y_hat, y)
                if gradient != 0:
                    gradient_norm.backward()
                    optimizer.step()
                    if gradient_norm < 1e-6:
                        losses.append(loss.item())
                        minimal_ratios.append(compute_minimal_ratio(gradient).item())

        progress_bar.set_description(f'Epoch: {epoch+1}/{epochs}')
    return losses, minimal_ratios

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

    repeat = 100
    all_losses = []
    all_minimum_ratios = []
    for i in range(repeat):
        # train the model
        losses, minimal_ratio = train_model_to_zero_gradient(model, train_loader, optimizer1, criterion, epochs)
        # sample 100 points from the losses
        if len(losses) > 100:
            indices = np.random.choice(len(losses), 100, replace=False)
            losses = np.array(losses)[indices]
            minimal_ratio = np.array(minimal_ratio)[indices]
        else:
            losses = np.array(losses)
            minimal_ratio = np.array(minimal_ratio)
        all_losses.append(losses)
        all_minimum_ratios.append(minimal_ratio)

    figure = plt.figure(figsize=(10, 10))
    # plot the minimal ratio and losses as points
    for i in range(repeat):
        plt.scatter(all_minimum_ratios[i], all_losses[i], label=f"Run {i+1}")
    plt.xlabel('Minimal Ratio')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Minimal Ratio vs Loss')
    plt.show() 
    figure.savefig('2-minimal_ratio_vs_loss.png')

