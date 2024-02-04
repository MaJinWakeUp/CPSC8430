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
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.main(x)

def train_first_stage(model, train_loader, optimizer, criterion, epochs=1000):
    model.train()
    losses = []
    gradients = []
    minimal_ratios = []
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
        progress_bar.set_description(f'Epoch: {epoch+1}/{epochs}')
    return losses, gradients, minimal_ratios

def epoch_loss(model, criterion, train_loader):
    with torch.no_grad():
        model.eval()
        losses = []
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            losses.append(loss.item())
        model.train()
    return sum(losses) / len(losses)


def train_second_stage(model, train_loader, optimizer, criterion, epochs=1000):
    model.train()
    losses = []
    gradients = []
    minimal_ratios = []
    progress_bar = tqdm(range(epochs))

    names = list(n for n, _ in model.named_parameters())
    def loss_hessian(params):
        y_hat = torch.func.functional_call(model, {n: p for n, p in zip(names, params)}, x)
        return criterion(y_hat, y)
    
    def closure():
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        return loss

    for epoch in progress_bar:
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            # optimizer.zero_grad()
            optimizer.step(closure)

        # Calculate gradient norm
        gradient = torch.cat([p.grad.flatten() for p in model.parameters()])
        gradient_norm = torch.norm(gradient, 2)
        if gradient_norm < 1e-2:
            losses.append(epoch_loss(model, criterion, train_loader))
            gradients.append(gradient_norm.item())
            hessian = torch.func.hessian(loss_hessian)(tuple(model.parameters()))
            num_pos_eig = 0
            num_all_eig = 0
            for hi in hessian:
                for tensor in hi:
                    # if not batches of square matrices, continue
                    if tensor.size(-1) != tensor.size(-2):
                        continue
                    eigenvalues = torch.linalg.eigvals(tensor)
                    positive_eigenvalues = (eigenvalues.real > 0).sum()
                    num_pos_eig += positive_eigenvalues
                    num_all_eig += eigenvalues.nelement()
            minimal_ratio = num_pos_eig / num_all_eig
            minimal_ratios.append(minimal_ratio.item())
            

        progress_bar.set_description(f'Epoch: {epoch+1}/{epochs}')

    return losses, gradients, minimal_ratios

if __name__ == "__main__":
    # Create the data
    x = torch.linspace(-10, 10, 10000).reshape(-1, 1)
    y = torch.sin(2 * torch.pi * x) / (2 * torch.pi * x)

    # parameters
    train_loader = torch.utils.data.DataLoader(list(zip(x, y)), batch_size=2000, shuffle=True)
    lr = 1e-3
    criterion = nn.MSELoss()
    epochs = 750

    repeat = 100
    all_losses = []
    all_minimum_ratios = []
    for i in range(repeat):
        model = DNN().to(device)
        optimizer1 = optim.Adam(model.parameters(), lr=lr)
        optimizer2 = optim.LBFGS(model.parameters(), lr=lr, history_size=5)
        # train the model with original criterion for 700 epochs
        _, _, _ = train_first_stage(model, train_loader, optimizer1, criterion, epochs)
        losses, gradients, minimal_ratio = train_second_stage(model, train_loader, optimizer2, criterion, epochs) 
        print(f'loss: {losses}, gradient: {gradients}, minimal_ratio: {minimal_ratio}')
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
        plt.scatter(all_minimum_ratios[i], all_losses[i])
    plt.xlabel('Minimal Ratio')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Minimal Ratio vs Loss')
    plt.show() 
    figure.savefig('./figs/2-minimal_ratio_vs_loss.png')

