import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from dataset import *
from models import *

import time

from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt
import numpy as np
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def cifar10():
    print('==> Preparing data...')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root='../tutorial/data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='../tutorial/data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    return trainset, trainloader, testset, testloader, classes

def main(args):
    summary = SummaryWriter()

    ### load model
    mod = __import__('models', fromlist=['*'])
    model = getattr(mod, args.model)

    ### Training on GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = model().to(device)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    # multiply LR by 1 / 10 after every 30 epochs
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # load dataset
    data = __import__('dataset', fromlist=['*'])
    dataset = getattr(data, args.dataset)
    dataset = dataset()

    ### checkpoint path
    dir_name = args.model + time.strftime('-%Y%m%d-%H%M%S', time.localtime(time.time()))
    checkpoint_path = os.path.join('checkpoints', dir_name)
    if not os.path.exists(checkpoint_path):
        print('creating dir {}'.format(checkpoint_path))
        os.mkdir(checkpoint_path)

    step = 0
    for epoch in range(2000):  # loop over the dataset multiple times
        lr_scheduler.step()
        running_loss = 0.0
        for i, data in enumerate(dataset.trainloader, 0):
            step += 1
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            #print(i)
            if step % 100 == 0:  # print every 100 mini-batches
                print("[Step: %10d] Loss: %f" % (step, loss))
                summary.add_scalar('loss', loss, step)
                # print('[%d, %5d] loss: %.3f' %
                #       (epoch + 1, i + 1, running_loss / 100))
                # summary.add_scalar('loss', loss, i)
                # running_loss = 0.0

        # save checkpoints
        checkpoint_file_path = os.path.join(checkpoint_path, 'epoch-{}.pkl'.format(epoch))
        print('==> Saving checkpoint... epoch {}'.format(epoch))
        state = {
            'epoch': epoch,
            'total_steps': step,
            'optimizer': optimizer.state_dict(),
            'model': net.state_dict(),
        }
        torch.save(state, checkpoint_file_path)
    print('==> Finished Training')

    ### 5. Test the network on the test data
    dataiter = iter(dataset.testloader)
    images, labels = dataiter.next()

    # print image
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % dataset.classes[labels[j]] for j in range(4)))

    outputs = net(images)

    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % dataset.classes[predicted[j]]
                                  for j in range(4)))

    ### performance of the network on the whole dataset
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataset.testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="arguments",
        epilog="Goodbye"
    )
    parser.add_argument('--model', type=str, choices=['LeNet', 'AlexNet', 'ResNet'])
    parser.add_argument('--dataset', type=str, choices=['CIFAR10', 'ImageNet'])
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    args, _ = parser.parse_known_args()
    main(args)
