import torch
from torchvision import datasets, transforms
import random

def random_noise_image():
    # fake_datasets = datasets.FakeData(1, (1, 28, 28), 10) # number of images, size of image, number of class
    img = torch.randn((1, 28, 28))
    return img

train_datasets = datasets.MNIST('./datasetsCollection', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

test_datasets = datasets.MNIST('./datasetsCollection', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

copied_train_datasets = datasets.MNIST('./datasetsCollection', train=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

copied_test_datasets = datasets.MNIST('./datasetsCollection', train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

def MNIST_copied_dataset():
    return copied_train_datasets

def MNIST_dataloader(batch_size, test_batch_size, **kwargs):
    
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=test_batch_size, shuffle=True, **kwargs)
    
    return train_loader, test_loader

def MNIST_random_image(inversion_class_number):

    idx = (copied_test_datasets.targets == inversion_class_number)

    copied_test_datasets.targets = copied_test_datasets.targets[idx]
    copied_test_datasets.data = copied_test_datasets.data[idx]

    image_idx = random.randrange(copied_test_datasets.__len__())

    return copied_test_datasets[image_idx][0], copied_test_datasets[image_idx][1]
