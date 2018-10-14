import torchvision.transforms as transforms

import config as cf

def transform_training():

    transform_train = transforms.Compose([
        transforms.Resize((cf.resize, cf.resize)),
        transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        # CIFAR10Policy(),
        transforms.ToTensor(),
    ])  # meanstd transformation

    return transform_train

def transform_testing():

    transform_test = transforms.Compose([
        transforms.Resize((cf.resize, cf.resize)),
        transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        # CIFAR10Policy(),
        transforms.ToTensor(),
    ])

    return transform_test
