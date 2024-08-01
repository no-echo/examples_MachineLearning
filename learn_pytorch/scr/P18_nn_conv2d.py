import torch
import torchvision
from sipbuild.generator import outputs
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../data/data_18", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

tudui = Tudui()

writer = SummaryWriter("../logs/logs_18")

step = 0
for data in dataloader:
    images, labels = data
    output = tudui(images)
    print(images.shape)
    print(output.shape)
    # torch.Size([64, 3, 32, 32])
    writer.add_images("input", images, step)
    # torch.Size([64, 6, 30, 30])

    output = torch.reshape(output, [-1, 3, 30, 30])
    writer.add_images("output", output, step)
    step += 1
