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
    trainloader1 = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader1 = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
    trainloader2 = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True)
    testloader2 = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False)

    model1 = CNN().to(device)
    model2 = CNN().to(device)
    model3 = CNN().to(device)
    optimizer1 = optim.Adam(model1.parameters(), lr=0.001)
    optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
    optimizer3 = optim.Adam(model3.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    accuracy = lambda y_hat, y: (y_hat.argmax(1) == y).float().mean()
    epochs = 100

    train_losses1, train_accuracies1, test_losses1, test_accuracies1 = train_model(model1, trainloader1, testloader1, optimizer1, criterion, accuracy, epochs)
    train_losses2, train_accuracies2, test_losses2, test_accuracies2 = train_model(model2, trainloader2, testloader2, optimizer2, criterion, accuracy, epochs)
    train_losses3, train_accuracies3, test_losses3, test_accuracies3 = train_model(model3, trainloader1, testloader1, optimizer3, criterion, accuracy, epochs)


    # Create a new model with interpolated weights
    # alpha from 0 to 1
    alphas = [i/100 for i in range(101)]
    new_train_losses_bs = []
    new_train_accuracies_bs = []
    new_test_losses_bs = []
    new_test_accuracies_bs = []
    new_train_losses_lr = []
    new_train_accuracies_lr = []
    new_test_losses_lr = []
    new_test_accuracies_lr = []
    for alpha in alphas:
        new_model_bs = CNN().to(device)
        for param1, param2, param_new in zip(model1.parameters(), model2.parameters(), new_model_bs.parameters()):
            param_new.data = (1 - alpha) * param1.data + alpha * param2.data

        new_optimizer = optim.Adam(new_model_bs.parameters(), lr=0.001)
        # Calculate loss using the new model
        new_train_loss, new_train_accuracy = train_epoch(new_model_bs, trainloader1, new_optimizer, criterion, accuracy, epochs)
        new_test_loss, new_test_accuracy = test_epoch(new_model_bs, testloader1, criterion, accuracy, epochs)
        new_train_losses_bs.append(new_train_loss)
        new_train_accuracies_bs.append(new_train_accuracy)
        new_test_losses_bs.append(new_test_loss)
        new_test_accuracies_bs.append(new_test_accuracy)

        new_model_lr = CNN().to(device)
        for param1, param2, param_new in zip(model1.parameters(), model3.parameters(), new_model_lr.parameters()):
            param_new.data = (1 - alpha) * param1.data + alpha * param2.data
        new_train_loss, new_train_accuracy = train_epoch(new_model_lr, trainloader1, new_optimizer, criterion, accuracy, epochs)
        new_test_loss, new_test_accuracy = test_epoch(new_model_lr, testloader1, criterion, accuracy, epochs)
        new_train_losses_lr.append(new_train_loss)
        new_train_accuracies_lr.append(new_train_accuracy)
        new_test_losses_lr.append(new_test_loss)
        new_test_accuracies_lr.append(new_test_accuracy)
    
    # Plot the results
    figure = plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(alphas, new_train_losses_bs, label='Train Loss', color='cyan')
    plt.plot(alphas, new_test_losses_bs, label='Test Loss', color='cyan', linestyle='--')
    plt.plot(alphas, new_train_accuracies_bs, label='Train Accuracy', color='orange')
    plt.plot(alphas, new_test_accuracies_bs, label='Test Accuracy', color='orange', linestyle='--')
    plt.xlabel('Alpha')
    plt.ylabel('Loss/Accuracy')
    plt.title('Interpolated Weights (Batch Size)')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(alphas, new_train_losses_lr, label='Train Loss', color='cyan')
    plt.plot(alphas, new_test_losses_lr, label='Test Loss', color='cyan', linestyle='--')
    plt.plot(alphas, new_train_accuracies_lr, label='Train Accuracy', color='orange')
    plt.plot(alphas, new_test_accuracies_lr, label='Test Accuracy', color='orange', linestyle='--')
    plt.xlabel('Alpha')
    plt.ylabel('Loss/Accuracy')
    plt.title('Interpolated Weights (Learning Rate)')
    plt.legend()
    figure.savefig('./figs/3-flatness_p1.png')