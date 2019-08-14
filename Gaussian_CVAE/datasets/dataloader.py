import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision
from torch.distributions import MultivariateNormal
import numpy as np

def load_mnist_data(BATCH_SIZE, model_kwargs):
    from torchvision import transforms
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transforms = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(
    './datasets/mnist_data',
    train=True,
    download=True,
    transform=transforms)

    test_dataset = datasets.MNIST(
        './datasets/mnist_data',
        train=False,
        download=True,
        transform=transforms
    )

    train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    return train_iterator, test_iterator
