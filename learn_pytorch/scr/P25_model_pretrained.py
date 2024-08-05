import torchvision
from torch import nn

# train_data = torchvision.datasets.ImageNet(root='../data/data_25', train=True, download=True,
#                                           transform=torchvision.transforms.ToTensor())

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)

print(vgg16_true)

train_data = torchvision.datasets.CIFAR10(root='../data/data_18', train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())

# vgg16_true.add_module('add_linear', nn.Linear(1000, 10))
vgg16_true.classifier.add_module('add_linear', nn.Linear(in_features=1000, out_features=10))
print(vgg16_true)

print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(in_features=4096, out_features=10)
print(vgg16_false)