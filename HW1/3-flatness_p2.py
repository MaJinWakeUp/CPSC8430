import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 14x14 -> 7x7
        self.fc1 = nn.Linear(7 * 7 * 32, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
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
    # Train 5 models with different batch sizes
    batch_sizes = [16, 32, 64, 128, 256, 512, 1024]
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    sensitivities = []

    for batch_size in batch_sizes:
        # Create a new model instance
        model = CNN().to(device)
        
        # Define the loss function, optimizer, and evaluation metric
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        accuracy = lambda y_hat, y: (y_hat.argmax(dim=1) == y).float().mean()
        
        # Create data loaders with the specified batch size
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        
        # Train the model and record the loss and accuracy
        train_loss, train_acc, test_loss, test_acc = train_model(model, train_loader, test_loader, optimizer, criterion, accuracy, epochs=100)
        train_losses.append(train_loss[-1])
        train_accuracies.append(train_acc[-1])
        test_losses.append(test_loss[-1])
        test_accuracies.append(test_acc[-1])
        
        # Calculate the sensitivity (Frobenius norm of gradients of loss to input)
        model.eval()
        sensitivity = 0
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            x.requires_grad = True
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            sensitivity += torch.norm(x.grad).item()
        sensitivities.append(sensitivity/len(test_loader))
        


    # plot the results
    figure = plt.figure(figsize=(10, 5))
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(batch_sizes, train_losses, label='Train', color='green')
    ax1.plot(batch_sizes, test_losses, label='Test', color='green', linestyle='dashed')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.plot(batch_sizes, sensitivities, label='Sensitivity', color='red')
    ax2.set_ylabel('Sensitivity')
    ax2.legend(loc='upper right')

    ax3 = plt.subplot(1, 2, 2)
    ax3.plot(batch_sizes, train_accuracies, label='Train', color='blue')
    ax3.plot(batch_sizes, test_accuracies, label='Test', color='blue', linestyle='dashed')
    ax3.set_xlabel('Batch Size')
    ax3.set_ylabel('Accuracy')
    ax3.legend(loc='upper left')
    ax4 = ax3.twinx()
    ax4.plot(batch_sizes, sensitivities, label='Sensitivity', color='red')
    ax4.set_ylabel('Sensitivity')
    ax4.legend(loc='upper right')

    figure.savefig('./figs/3-flatness_p2.png')