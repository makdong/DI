import torch
from torchvision import datasets, transforms

def MNIST_dataset(batch_size, test_batch_size, kwargs):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/MNIST', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/MNIST', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=test_batch_size, shuffle=True, **wargs)
    
    (train_loader, test_loader)