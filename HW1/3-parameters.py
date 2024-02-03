import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class CNN(nn.Module):
    def __init__(self, latent_dim=128):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(16, latent_dim, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 14x14 -> 7x7
        self.conv3 = nn.Conv2d(latent_dim, 16, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(7 * 7 * 16, 128)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        return x

def train_epoch(model, train_loader, optimizer, criterion, eval, epoch):
    model.train()
    losses = []
    accuracies = []
    progress_bar = tqdm(train_loader)
    for i, (x, y) in enumerate(progress_bar):
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)
        loss = criterion(y_hat, y)
        acc = eval(y_hat, y)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        accuracies.append(acc.item())
        progress_bar.set_description(f'Epoch: {epoch+1} training. Loss: {loss.item():.5f}, Accuracy: {acc.item():.5f}')
    loss = sum(losses) / len(losses)
    acc = sum(accuracies) / len(accuracies)
    return loss, acc

def test_epoch(model, test_loader, criterion, eval, epoch):
    model.eval()
    losses = []
    accuracies = []
    progress_bar = tqdm(test_loader)
    with torch.no_grad():
        for i, (x, y) in enumerate(progress_bar):
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            acc = eval(y_hat, y)

            losses.append(loss.item())
            accuracies.append(acc.item())
            progress_bar.set_description(f'Epoch: {epoch+1} testing. Loss: {loss.item():.5f}, Accuracy: {acc.item():.5f}')
    loss = sum(losses) / len(losses)
    acc = sum(accuracies) / len(accuracies)
    return loss, acc

def train_model(model, train_loader, test_loader, optimizer, criterion, eval, epochs=100):
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, eval, epoch)
        test_loss, test_acc = test_epoch(model, test_loader, criterion, eval, epoch)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
    return train_losses, train_accuracies, test_losses, test_accuracies

if __name__=="__main__":
    # Load the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    # Randomly change training and testing labels
    trainset.targets = torch.tensor([random.randint(0, 9) for _ in range(len(trainset.targets))])
    testset.targets = torch.tensor([random.randint(0, 9) for _ in range(len(testset.targets))])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

    # Create models
    latent_dims = [16+16*i for i in range(20)]
    models = [CNN(latent_dim=latent_dim).to(device) for latent_dim in latent_dims]
    model_parameters = [sum(p.numel() for p in model.parameters()) for model in models]
    final_train_losses = []
    final_test_losses = []
    final_train_accuracies = []
    final_test_accuracies = []
    for model in models:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        accuracy = lambda y_hat, y: (y_hat.argmax(dim=1) == y).float().mean()
        train_losses, train_accuracies, test_losses, test_accuracies = train_model(model, trainloader, testloader, optimizer, criterion, accuracy, epochs=100)
        final_train_losses.append(train_losses[-1])
        final_test_losses.append(test_losses[-1])
        final_train_accuracies.append(train_accuracies[-1])
        final_test_accuracies.append(test_accuracies[-1])

    figure = plt.figure(figsize=(10, 5))
    plt.add_subplot(1, 2, 1)
    plt.scatter(model_parameters, final_train_losses, label='Training Loss')
    plt.scatter(model_parameters, final_test_losses, label='Test Loss')
    plt.xlabel('Number of Parameters')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')
    plt.add_subplot(1, 2, 2)
    plt.scatter(model_parameters, final_train_accuracies, label='Training Accuracy')
    plt.scatter(model_parameters, final_test_accuracies, label='Test Accuracy')
    plt.xlabel('Number of Parameters')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')
    figure.savefig('./figs/3-parameters_vs_generalization.png')
    
