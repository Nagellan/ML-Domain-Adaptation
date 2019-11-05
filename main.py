##
# 1) Make better random seed: https://discuss.pytorch.org/t/random-seed-initialization/7854
##

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import numpy as np
import matplotlib.pyplot as plt

TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 100
DATA_FOLDER = "data"
SEED = 228
PRINT_LOGS = True

# Remove randomness by adding global uniform seed where needed
# Start of code snippet (1)
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def _init_fn():
    np.random.seed(SEED)
# End of code snippet (1)


# Transformations
svhn_transformations = transforms.Compose([
    transforms.Resize(28),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist_transformations = transforms.Compose([
    transforms.ToTensor(),
    # transforms.ColorJitter(brightness=0, contrast=0,
    transforms.Normalize((0.1307,), (0.3081,))
])

# Data Source
svhn_train = datasets.SVHN(DATA_FOLDER, download=True, split="train", transform=svhn_transformations)
svhn_test = datasets.SVHN(DATA_FOLDER, download=True, split="test", transform=svhn_transformations)
mnist_test = datasets.MNIST(DATA_FOLDER, download=True, train=False, transform=mnist_transformations)

# Data loaders
train_svhn_loader = DataLoader(svhn_train, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                               num_workers=0, worker_init_fn=_init_fn)
test_svhn_loader = DataLoader(svhn_test, batch_size=TEST_BATCH_SIZE, shuffle=True,
                              num_workers=0, worker_init_fn=_init_fn)
test_mnist_loader = DataLoader(mnist_test, batch_size=TEST_BATCH_SIZE, shuffle=True,
                               num_workers=0, worker_init_fn=_init_fn)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_bn = nn.BatchNorm2d(20)  # added Batch Norm
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model_cnn = Net().to(device)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            if PRINT_LOGS:
                print('Train Epoch: {} [{:05d}/{:05d} ({:02.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, set_name):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    if PRINT_LOGS:
        print('{}: Average loss: {:.4f}, Accuracy: {:05d}/{:05d} ({:.2f}%)'.format(
            set_name, test_loss, correct, len(test_loader.dataset), accuracy))

    return round(accuracy, 3)


epochs = 10
lr = 0.005
momentum = 0.5
log_interval = 700

model = model_cnn
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
# optimizer = optim.ASGD(model.parameters(), lr=lr)

epoch_list = list(range(1, epochs + 1))
svhn_train_accuracy = []
svhn_test_accuracy = []
mnist_accuracy = []

for epoch in epoch_list:
    train(model, device, train_svhn_loader, optimizer, epoch)
    svhn_train_accuracy.append(test(model, device, train_svhn_loader, "SVHN Train set"))
    svhn_test_accuracy.append(test(model, device, test_svhn_loader, "SVHN Test  set"))
    mnist_accuracy.append(test(model, device, test_mnist_loader, "MNIST Test set"))
    if PRINT_LOGS:
        print()
    torch.save(model.state_dict(), "mnist_inno.pt")

if PRINT_LOGS:
    x = np.arange(len(epoch_list))  # the label locations
    width = 0.23  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, svhn_train_accuracy, width, label="SVHN Train")
    rects2 = ax.bar(x, svhn_test_accuracy, width, label="SVHN Test")
    rects3 = ax.bar(x + width, mnist_accuracy, width, label="MNIST")
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Epochs")
    ax.set_title("SVHN and MNIST sets accuracy over multiple epochs")
    ax.set_xticks(x)
    ax.set_xticklabels(epoch_list)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    fig.tight_layout()
    plt.margins(y=0.1)
    plt.show()

print(mnist_accuracy[-1])
