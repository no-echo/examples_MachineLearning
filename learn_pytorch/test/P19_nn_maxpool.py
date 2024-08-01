import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='./data_18', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        output = self.maxpool(input)
        return output

tudui = Tudui()

writer = SummaryWriter('../logs/logs_maxpool')
step = 0

for data in dataloader:
    imgs, targets = data
    writer.add_images('input', imgs, step)
    outputs = tudui(imgs)
    writer.add_images('outputs', outputs, step)
    step += 1

writer.close()