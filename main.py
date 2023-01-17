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
import os
import sys
from PIL import Image

from datasetsCollection import datasetsCollection
from models import mnist
from utils import logger

# set the directory
default_dir = os.path.abspath("/home/dhkim0317/DI")
os.chdir(default_dir)

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

    def inverse():
    
        return

    
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
                        help='number for k-nn method.')
    parser.add_argument('--inversion_class_number', type=int, default='-1',
                        help='number of class to invert')
    parser.add_argument('--random_init', action='store_true', default='false',
                        help='set true for random initialization.')
    parser.add_argument('--inversion_epochs', type=int, default='300',
                        help='number of epochs to update the input image.')
    args = parser.parse_args()

    logger.set_logfile_name(args.log_name)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    model = mnist.MNISTNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    if args.datasets == 'MNIST':
        train_loader, test_loader = datasetsCollection.MNIST_dataloader(args.batch_size, args.test_batch_size, **kwargs)
    else:
        train_loader, test_loader = datasetsCollection.MNIST_dataloader(args.batch_size, args.test_batch_size, **kwargs)

    if (args.model_loaded is False):
        logger.logger.info('Start training model')

        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            test(args, model, device, test_loader)
        
        logger.logger.info('Finish training model')

        if (args.save_model):
            torch.save(model.state_dict(), "./weights/" + args.model_save_name)
        
        logger.logger.info('Finish Saving model')
    else:
        logger.logger.info('Model already exist')
        model.load_state_dict(torch.load("./weights/" + args.model_save_name))
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        logger.logger.info('Model loaded')
    
    logger.logger.info('Finish model working')

    logger.logger.info('Start inversion')

    class_num = 10

    # input image, output image 존재
    # output image = target output
    # input image = random 이거나 k-NN

    inversion_class_number = args.inversion_class_number
    if (inversion_class_number == -1):
        logger.logger.info('random class number')
        inversion_class_number = random.randrange(class_num)
        logger.logger.info('random class number is {}'.format(inversion_class_number))
    elif (inversion_class_number >= 10 or inversion_class_number < 0):
        logger.logger.error('wrong input : class number')
        exit()
    else:
        logger.logger.info('class number is {}'.format(inversion_class_number))
    
    logger.logger.info('Target image is selected')
    target_image, target_label = datasetsCollection.MNIST_random_image(inversion_class_number)

    copied_train_datasets = datasetsCollection.MNIST_copied_dataset()
    copied_train_images = list(map(lambda x : x[0], copied_train_datasets))
    # copied_train_labels = list(map(lambda x: x[1], copied_train_datasets))
    copied_train_images_distribution = list(map(model, copied_train_images))
    
    target_output_distribution = model(target_image)
    
    distribution_target_distances = np.array([])    
    for distribution in copied_train_images_distribution:
        distribution_target_distances = np.append(distribution_target_distances, F.pairwise_distance(target_output_distribution.detach(), distribution.detach()))
        # distribution_target_distances.append(F.pairwise_distance(target_output_distribution.detach(), distribution.detach()))

    # distribution_target_distances = np.array(distribution_target_distances)

    if args.random_init == True:
        logger.logger.info('random initialization.')
        input_image = datasetsCollection.random_noise_image
        # input_label = inversion_class_number
    else:
        logger.logger.info('K-NN initialization.')
        index_set = set()
        nn_sum = torch.zeros(target_output_distribution.shape)

        for idx in range(0, args.k):
            for nn_index in index_set:
                distribution_target_distances[nn_index] = 1
            
            nearest_idx = distribution_target_distances.argmin()
            index_set.add(nearest_idx)
            
            nn_sum += copied_train_images_distribution[nearest_idx]
        
        input_image = nn_sum / args.k
    
    velocity = {}

    for layer in model.state_dict():
        velocity[layer] = np.zeros(model.state_dict()[layer].shape)

    logger.logger.info('inversion start')

    model.eval()

    for epoch in range(0, args.inversion_epochs):
        avg_model_output = model.inversion_forward(input_image)

        loss = F.nll_loss(avg_model_output, target_output_distribution)
        loss.backward()
        
        velocity = args.inversion_momentum * velocity - args.inversion_lr * input_image.grad

        input_image = input_image + velocity
        
        if (epoch % args.log_interval == 0):
            logger.logger.info('Inversion Epoch: {}'.format(
                epoch
            ))

    logger.logger.info("Saving Image...")
    
    transform = transforms.ToPILImage()
    img = transform(input_image)

    img.save("output_image.png")
    
    logger.logger.info("Saving Image Done")

    logger.logger.info("Every work is done")

if __name__ == '__main__':
    main()


# target output vector t, number of nearest neighbors k, Distance measure d()
# t와 output distribution이 가장 비슷한 k개의 image의 output distribution의 average를 사용
# 유사도를 측정할 때 d()을 사용
# k개의 image는 training dataset에서 차출 가능, 이 때 label로 미리 거르는 내용은 논문에는 없긴 한데 있어도 될 듯
# t는 어디서 구하나? image of specific number를 trained DNN에 pass한 뒤의 distribution인데, image of specific number를 어디서 가져와야 하나?
# 만약 이 또한 training image에서 갖고 오는 거라면, nearest neighbor를 구할 때에는 이걸 배제하고 찾아야 하나?
# target output t는 test dataset에서 가져오도록 하자.

# 찾게 되면, 논문 속 loss function을 구현하고 300번동안 gradient descent 적용해서 input image를 구함
# 최초 input이 average of k-nn 이고, input을 새로 계산함
# model의 weight는 그대로 유지하고, I_new = I_old + learning_rate * gradient_descent_(I_i)(I) + momentum * gradient_descent_(I_i)(I_old)