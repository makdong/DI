import torch
from torchvision import datasets, transforms
import random

train_datasets = datasets.MNIST('../datasetsCollection', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

test_datasets = datasets.MNIST('../datasetsCollection', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

def MNIST_dataset(batch_size, test_batch_size, **kwargs):
    
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=test_batch_size, shuffle=True, **kwargs)
    
    return train_loader, test_loader

def MNIST_random_image():
    image_idx = random.randrange(train_datasets.__len__())

    return train_datasets[image_idx][0], train_datasets[image_idx][1]
