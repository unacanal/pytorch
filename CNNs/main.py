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

def main(args):

    ### load model
    mod = __import__('models', fromlist=['*'])
    model = getattr(mod, args.model)

    ### tensorboard
    summary = SummaryWriter(comment=args.model)

    ### Training on GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = model().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.Adam(net.parameters(), lr=args.lr)

    # multiply LR by 1 / 10 after every 30 epochs FOR ALEXNET
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # load dataset
    data = __import__('dataset', fromlist=['*'])
    dataset = getattr(data, args.dataset)
    dataset = dataset(args)

    ### checkpoint path
    dir_name = args.model + time.strftime('-%Y%m%d-%H%M%S', time.localtime(time.time()))
    checkpoint_path = os.path.join('checkpoints', dir_name)
    if not os.path.exists(checkpoint_path):
        print('creating dir {}'.format(checkpoint_path))
        os.mkdir(checkpoint_path)

    if args.model == 'VGG19':
        vgg11 = torch.load('./checkpoints/VGG11-20200408-212150/epoch-99.pkl')
        pretrained_dict = vgg11.state_dict()
        net_dict = net.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}
        # 2. overwrite entries in the existing state dict
        net_dict.update(pretrained_dict)
        # 3. load the new state dict
        net.load_state_dict(net_dict)

        print("==> Load pre-trained weights...")

    step = 0
    for epoch in range(100):  # loop over the dataset multiple times
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
            if step % 100 == 0:
                print("[Step: %10d] Loss: %f" % (step, loss))
                summary.add_scalar('train_loss', loss, step)

        # save checkpoints
        checkpoint_file_path = os.path.join(checkpoint_path, 'epoch-{}.pkl'.format(epoch))
        print('==> Saving checkpoint... epoch {}'.format(epoch))
        torch.save(net, checkpoint_file_path)
    print('==> Finished Training')

    ### 5. Test the network on the test data
    # dataiter = iter(dataset.testloader)
    # images, labels = dataiter.next()

    # # print image
    # imshow(torchvision.utils.make_grid(images))
    # print('GroundTruth: ', ' '.join('%5s' % dataset.classes[labels[j]] for j in range(4)))
    #
    # outputs = net(images)
    #
    # _, predicted = torch.max(outputs, 1)
    # print('Predicted: ', ' '.join('%5s' % dataset.classes[predicted[j]]
    #                               for j in range(4)))

    ### performance of the network on the whole dataset
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataset.testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %.3f %%' % (
            100 * correct / total))

    ### what are the classes that perform well?
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in dataset.testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(args.batch):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %.3f %%' % (
            dataset.classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="arguments",
        epilog="Goodbye"
    )
    parser.add_argument('--model', type=str, choices=['LeNet', 'AlexNet', 'AlexNet2', 'VGG11', 'VGG19', 'ResNet'])
    parser.add_argument('--dataset', type=str, choices=['CIFAR10', 'ImageNet'])
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--batch', type=int, default=128, help='batch size')
    args, _ = parser.parse_known_args()
    main(args)
