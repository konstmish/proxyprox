import numpy as np
import os
import random
import torch
import torchvision

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from filelock import FileLock


def load_data(dataset='cifar10', batch_size=128, num_workers=4):
    """
    Loads the required dataset
    :param dataset: Can be either 'cifar10' or 'cifar100'
    :param batch_size: The desired batch size
    :return: Tuple (train_loader, test_loader, num_classes)
    """
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    mnist_transforms = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307, ), (0.3081, ))])

    if dataset == 'cifar10':
        # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        num_classes = 10
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    elif dataset == 'cifar100':
        num_classes = 100
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    elif dataset == 'mnist':
        num_classes = 10
        with FileLock(os.path.expanduser("~/data.lock")):
            trainset = torchvision.datasets.MNIST(
                    "~/data",
                    train=True,
                    download=True,
                    transform=mnist_transforms)
            testset = torchvision.datasets.MNIST("~/data", train=False, transform=mnist_transforms)
    else:
        raise ValueError('Only cifar 10, cifar 100, and MNIST are supported')

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader, num_classes


def test_accuracy(net, testloader, device):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def accuracy_and_loss(net, dataloader, device, criterion):
    net.eval()
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss += criterion(outputs, labels).cpu().item() / len(dataloader)

    return correct / total, loss
            
            
def seed_everything(seed=1029):
    '''
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
