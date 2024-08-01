import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='../data/data_18', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.sigmoid(input)
        return output

tudui = Tudui()
step = 0
writer = SummaryWriter(log_dir='../logs/logs_20')
for data in dataloader:
    imgs, labels = data
    writer.add_images('input', imgs, step)
    outputs = tudui(imgs)
    writer.add_images('outputs', outputs, step)
    step += 1

writer.close()
