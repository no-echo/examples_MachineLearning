import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),

])
trans_set = torchvision.datasets.CIFAR10(root="../data/dataset_11", train=True, transform=dataset_transforms, download=True)
test_set = torchvision.datasets.CIFAR10(root="../data/dataset_11", train=False, transform=dataset_transforms, download=True)

writer = SummaryWriter("../logs/p10")
for i in range(10):
    img, target = trans_set[i]
    writer.add_image("test_set", img, i)

writer.close()