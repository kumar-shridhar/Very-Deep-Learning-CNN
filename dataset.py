import torch
import torchvision
import sys
from transform import transform_training, transform_testing
import config as cf

def dataset(dataset_name):

    if (dataset_name == 'cifar10'):
        print("| Preparing CIFAR-10 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_training())
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_testing())
        outputs = 10
        inputs = 3
    
    elif (dataset_name == 'cifar100'):
        print("| Preparing CIFAR-100 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_training())
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_testing())
        outputs = 100
        inputs = 3
    
    elif (dataset_name == 'mnist'):
        print("| Preparing MNIST dataset...")
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_training())
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform_testing())
        outputs = 10
        inputs = 1
    
    elif (dataset_name == 'fashionmnist'):
        print("| Preparing FASHIONMNIST dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_training())
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform_testing())
        outputs = 10
        inputs = 1
    elif (dataset_name == 'stl10'):
        print("| Preparing STL10 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.STL10(root='./data',  split='train', download=True, transform=transform_training())
        testset = torchvision.datasets.STL10(root='./data',  split='test', download=False, transform=transform_testing())
        outputs = 10
        inputs = 3
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=cf.batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=cf.batch_size, shuffle=False, num_workers=4)
    
    return trainloader, testloader, outputs, inputs

