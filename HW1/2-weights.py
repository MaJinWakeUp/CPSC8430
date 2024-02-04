import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import torch.optim as optim
import torchvision

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

def train_epoch(model, train_loader, optimizer, criterion, eval, epoch):
    model.train()
    losses = []
    accuracies = []
    progress_bar = tqdm(train_loader)
    for i, (x, y) in enumerate(progress_bar):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
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

def train_model(model, train_loader, test_loader, optimizer, criterion, eval, epochs=100, vis=True, vis_interval=1):
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    layer_weights = []
    model_weights = []
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, eval, epoch)
        test_loss, test_acc = test_epoch(model, test_loader, criterion, eval, epoch)
        if vis and epoch % vis_interval == 0:
            # if vis, only collect weights, loss, accuracy every k epochs
            layer_weights.append(model.fc1.weight.view(-1).cpu().detach().numpy())
            # print(layer_weights[-1].shape)
            model_weights.append(torch.cat([p.view(-1) for p in model.parameters()]).cpu().detach().numpy())
            # print(model_weights[-1].shape)
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)
        else:
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)
    return train_losses, train_accuracies, test_losses, test_accuracies, layer_weights, model_weights

if __name__=="__main__":
    # Load the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    accuracy = lambda y_hat, y: (y_hat.argmax(dim=1) == y).float().mean()
    epochs = 100
    vis = True
    vis_interval = 5

    all_train_losses = []
    all_train_accuracies = []
    all_layer_weights = []
    all_model_weights = []
    # Train the model
    repeat = 8
    for i in range(repeat):
        model = DNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_losses, train_accuracies, test_losses, test_accuracies, layer_weights, model_weights = train_model(model, trainloader, testloader, optimizer, criterion, accuracy, epochs, vis, vis_interval)
        # Convert weights to two-dimensional points using PCA
        layer_weights_pca = PCA(n_components=2).fit_transform(np.array(layer_weights))
        model_weights_pca = PCA(n_components=2).fit_transform(np.array(model_weights))
        all_train_losses.append(train_losses)
        all_train_accuracies.append(train_accuracies)
        all_layer_weights.append(layer_weights_pca)
        all_model_weights.append(model_weights_pca)
    
    # save values first before plotting
    np.save('2-all_train_losses.npy', all_train_losses)
    np.save('2-all_train_accuracies.npy', all_train_accuracies)
    np.save('2-all_layer_weights.npy', all_layer_weights)
    np.save('2-all_model_weights.npy', all_model_weights)
    # Visualize loss value with respect to the weights in 3D, while weights are x and y, and loss is z
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    for i in range(repeat):
        ax.plot(all_layer_weights[i][:, 0], all_layer_weights[i][:, 1], all_train_losses[i])
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_zlabel('Loss')
    plt.legend()
    plt.title('Loss with respect to layer weights')

    ax = fig.add_subplot(122, projection='3d')
    for i in range(repeat):
        ax.plot(all_model_weights[i][:, 0], all_model_weights[i][:, 1], all_train_losses[i])
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_zlabel('Loss')
    plt.legend()
    plt.title('Loss with respect to model weights')

    plt.show()
    fig.savefig('./figs/2-weights.png')



