import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class CNN1(nn.Module):
    def __init__(self):
        super(CNN1, self).__init__()
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

class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 14x14 -> 7x7
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
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
    
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.main(x)

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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

    # Create models
    cnn1 = CNN1().to(device)
    cnn2 = CNN2().to(device)
    dnn = DNN().to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer1 = optim.Adam(cnn1.parameters(), lr=0.001)
    optimizer2 = optim.Adam(cnn2.parameters(), lr=0.001)
    optimizer3 = optim.Adam(dnn.parameters(), lr=0.001)
    accuracy = lambda y_hat, y: (y_hat.argmax(dim=1) == y).float().mean()
    epochs = 100

    # Train the models
    train_losses1, train_accuracies1, test_losses1, test_accuracies1 = train_model(cnn1, trainloader, testloader, optimizer1, criterion, accuracy, epochs)
    train_losses2, train_accuracies2, test_losses2, test_accuracies2 = train_model(cnn2, trainloader, testloader, optimizer2, criterion, accuracy, epochs)
    train_losses3, train_accuracies3, test_losses3, test_accuracies3 = train_model(dnn, trainloader, testloader, optimizer3, criterion, accuracy, epochs)

    # Plot the training and test loss curves
    figure1 = plt.figure(figsize=(10, 5))
    plt.plot(train_losses1, label='Train loss (CNN1)', color='cyan')
    plt.plot(test_losses1, label='Test loss (CNN1)', color='cyan', linestyle='--')
    plt.plot(train_losses2, label='Train loss (CNN2)', color='orange')
    plt.plot(test_losses2, label='Test loss (CNN2)', color='orange', linestyle='--')
    plt.plot(train_losses3, label='Train loss (DNN)', color='green')
    plt.plot(test_losses3, label='Test loss (DNN)', color='green', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Curves')
    plt.legend()
    figure1.savefig('1-mnist_loss.png')
    plt.close()

    # Plot the training and test accuracy curves
    figure2 = plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies1, label='Train accuracy (CNN1)', color='cyan')
    plt.plot(test_accuracies1, label='Test accuracy (CNN1)', color='cyan', linestyle='--')
    plt.plot(train_accuracies2, label='Train accuracy (CNN2)', color='orange')
    plt.plot(test_accuracies2, label='Test accuracy (CNN2)', color='orange', linestyle='--')
    plt.plot(train_accuracies3, label='Train accuracy (DNN)', color='green')
    plt.plot(test_accuracies3, label='Test accuracy (DNN)', color='green', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Test Accuracy Curves')
    plt.legend()
    figure2.savefig('1-mnist_accuracy.png')
    plt.close()
