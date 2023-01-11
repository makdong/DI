import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import datetime
import random

from datasetsCollection import datasetsCollection
from models import mnist
from utils import logger

def main():
    def train(args, model, device, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                logger.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    def test(args, model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        logger.logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    # Training settings
    current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    parser = argparse.ArgumentParser(description='DI Lab Research Participation 2022 Winter')

    parser.add_argument('--log_name', type=str, default=current_time,
                        help='choose the name of log file')
    parser.add_argument('--model', type=str, default=None,
                        help='select the model')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--inversion_lr', type=float, default=0.1, metavar='I-LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--inversion_momentum', type=float, default=0.9, metavar='I-M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--model_loaded', action='store_true', default=False,
                        help='does model already exist or not')
    parser.add_argument('--model_save_name', type=str, default='mnist_cnn.pt',
                        help='Name for saving model weights')
    parser.add_argument('--datasets', type=str, default='MNIST',
                        help='Datasets for model')
    parser.add_argument('--k', type=int, default='3',
                        help='number for k-nn method')
    parser.add_argument('--nn_index', type=int, default='10',
                        help='number for k-nn neighbors')
    args = parser.parse_args()

    logger.set_logfile_name(args.log_name)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    model = mnist.MNISTNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    if args.datasets == 'MNIST':
        train_loader, test_loader = datasetsCollection.MNIST_dataset(args.batch_size, args.test_batch_size, **kwargs)
    else:
        train_loader, test_loader = datasetsCollection.MNIST_dataset(args.batch_size, args.test_batch_size, **kwargs)

    if (args.model_loaded is False):
        logger.logger.info('Start training model')

        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            test(args, model, device, test_loader)
        
        logger.logger.info('Finish training model')

        if (args.save_model):
            torch.save(model.state_dict(), "../weights/" + args.model_save_name)
        
        logger.logger.info('Finish Saving model')
    else:
        logger.logger.info('Model already exist')
        model.load_state_dict(torch.load("../weights/" + args.model_save_name))
        logger.logger.info('Model loaded')
    
    logger.logger.info('Finish model working')

    logger.logger.info('Start inversion')

    class_num = 10
    inversion_class = random.randrange(class_num)

    # model.predict_class

    image, label = datasetsCollection.MNIST_random_image
    
    target_output_distribution = model.forward(image)
    copy_train_loader, copy_test_loader = datasetsCollection.MNIST_dataset(args.batch_size, args.test_batch_size, **kwargs)
    index_set = set()

    forwarded_copy_train_loader = list(map(model.forward(), copy_train_loader))
    
    F.pairwise_distance(target_output_distribution, target_output_distribution)





if __name__ == '__main__':
    main()


# 1. 학습된 model --> done
# 2. target output 제공 -> target output은 specific class가 아닌 output distribution을 준다. (specific class는 미분이 안되서)
# 2-ex. image of specific number를 trained DNN에 pass하고 난 distribution을 target output으로 사용
# 3. inversion을 통해 target output을 만들 것 -> 가장 효과 좋은 init image를 만드는 게 목표