import torch
import torchvision
import torchvision.transforms as transforms

class CIFAR10:
    def __init__(self, args)
        self.args = args
        print('==> Preparing data...')
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        self.trainset = torchvision.datasets.CIFAR10(root='../tutorial/data', train=True,
                                            download=True, transform=self.transform_train)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=16,
                                              shuffle=True, num_workers=2)

        self.testset = torchvision.datasets.CIFAR10(root='../tutorial/data', train=False,
                                           download=True, transform=self.transform_test)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=16,
                                             shuffle=False, num_workers=2)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck')