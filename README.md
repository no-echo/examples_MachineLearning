```

```

# pytorch学习

## pytorch学习的两大法宝

+ dir( ):打开，看见里边有什么
+ help( ):说明书

## python编译器的比较

### python文件

+ 通用，传播方便，适用于大型项目
+ 需要从头运行

### python控制台

+ 以任意行为块运行，shift + enter可以多行输入，按 ⬆️可以返回上段代码
+ 显示每个变量的属性，便与调试
+ 不利于代码修改和阅读

### Jupyter

+ 利于代码的阅读和修改
+ 环境需要配置

## 如何读取数据-两种方式

### Dataset

+ 提供一种方式去获取数据及其label值
+ 实现：如何获取每一个数据及其label，告诉我们总共有多少个数据
+ 实例

```python
from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)

root_dir = 'dataset/train'
ants_label_dir = 'ants'
bees_label_dir = 'bees'
ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)

train_dataset = ants_dataset + bees_dataset
```



```python
from torch.utils.data import Dataset  # 导入 PyTorch 的 Dataset 基类
from PIL import Image  # 导入 PIL 库用于图像处理
import os  # 导入 os 库用于文件和目录操作

# 定义一个继承自 Dataset 的自定义数据集类 MyData
class MyData(Dataset):
    
    # 初始化方法，传入根目录和标签目录
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir  # 根目录
        self.label_dir = label_dir  # 标签目录
        self.path = os.path.join(self.root_dir, self.label_dir)  # 图片文件夹的完整路径
        self.img_path = os.listdir(self.path)  # 获取图片文件夹中的所有文件名

    # 获取数据集中的一个样本
    def __getitem__(self, idx):
        img_name = self.img_path[idx]  # 获取第 idx 个图片文件的文件名
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)  # 构建图片文件的完整路径
        img = Image.open(img_item_path)  # 打开图片文件
        label = self.label_dir  # 标签即为图片所在的子文件夹名称
        return img, label  # 返回图片和标签

    # 返回数据集中样本的数量
    def __len__(self):
        return len(self.img_path)  # 返回图片文件的数量

# 定义根目录和标签目录
root_dir = 'dataset/train'  # 数据集根目录
ants_label_dir = 'ants'  # 蚂蚁图片文件夹
bees_label_dir = 'bees'  # 蜜蜂图片文件夹

# 创建蚂蚁数据集实例 
ants_dataset = MyData(root_dir, ants_label_dir)

# 创建蜜蜂数据集实例
bees_dataset = MyData(root_dir, bees_label_dir)

# 将两个数据集合并为一个训练集
train_dataset = ants_dataset + bees_dataset
```

#### Tensorboard的使用

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

# writer.add_image()
# y = 2x
for i in range(100):
    writer.add_scalar("y = 2x", 2*i, i)

writer.close()

'''
终端启动：
tensorboard --logdir=logs
打开浏览器，访问地址 http://localhost:6006/
通过这种方式，你可以使用 TensorBoard 轻松地可视化训练过程中的各种标量数据，帮助你更好地理解模型的训练动态。
'''
```

```python
from torch.utils.tensorboard import SummaryWriter  
# 从 PyTorch 的 tensorboard 模块中导入 SummaryWriter 类

# 创建一个 SummaryWriter 对象，并指定日志目录为 "logs"
writer = SummaryWriter("logs")

# writer.add_image()  # 这行被注释掉，提示可以使用 writer.add_image() 方法记录图像数据
# y = 2x  # 注释，指出接下来将记录 y = 2x 这条直线

# 循环遍历 0 到 99，用于生成数据
for i in range(100):
    # 使用 add_scalar 方法向日志中添加标量数据
    # 参数 "y = 2x" 是标量数据的标签名称
    # 2 * i 是记录的标量值，即 y = 2x 中的 y 值
    # i 是步数（step），通常用于表示当前记录的步骤数，在 TensorBoard 中用于 x 轴
    writer.add_scalar("y = 2x", 2 * i, i)

# 关闭 SummaryWriter 对象，确保所有数据都已写入磁盘
writer.close()
```

```python
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
image_path = "data/train/ants_image/5650366_e22b7e1065.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)

writer.add_image("test", img_array, 2, dataformats='HWC')
# y = 2
for i in range(100):
    writer.add_scalar("y = 2x", 2 * i, i)
writer.close()
```

```###python
from torch.utils.tensorboard import SummaryWriter  # 从 PyTorch 的 tensorboard 模块中导入 SummaryWriter 类
import numpy as np  # 导入 NumPy 库，用于数值计算和处理多维数组
from PIL import Image  # 从 PIL（Python Imaging Library）中导入 Image 模块，用于图像处理

# 创建一个 SummaryWriter 对象，并指定日志存放目录为 "logs"
writer = SummaryWriter("logs")

# 指定图像的路径
image_path = "data/train/ants_image/5650366_e22b7e1065.jpg"

# 使用 PIL 库打开图像，返回一个 PIL 图像对象
img_PIL = Image.open(image_path)

# 将 PIL 图像对象转换为 NumPy 数组，以便进行进一步的处理
img_array = np.array(img_PIL)

# 向日志中添加一张图像
# 参数 "test" 是这张图像的标签名称
# img_array 是要记录的图像数据
# 2 是步数（step），可以用来表示在训练的哪个步骤记录了该数据
# dataformats='HWC' 指定数据格式为 [Height, Width, Channels]，即图像是高宽通道顺序的彩色图像
writer.add_image("test", img_array, 2, dataformats='HWC')

# y = 2x 的图表注释
# 创建一个循环，从 0 到 99 用于生成 y = 2x 的数据点
for i in range(100):
    # 使用 add_scalar 方法将标量数据添加到日志中
    # 参数 "y = 2x" 是标量数据的标签名称
    # 2 * i 是标量的值，即 y = 2x 公式中的 y 值
    # i 是步数（step），用于表示当前记录的步骤数
    writer.add_scalar("y = 2x", 2 * i, i)

# 关闭 SummaryWriter 对象，确保所有数据都已写入到日志文件中
writer.close()
```

#### Transforms的使用-图像的变化

+ Transforms.py相当于一个工具箱，常用的工具：totensor, resize
+ 拿一些特定格式的图片 -> 经过transforms工具的加工 -> 输出一个结果

```python
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path = "data/train/ants_image/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("Tensor_img", tensor_img)

writer.close()
```

```python
from PIL import Image  # 从 PIL（Python Imaging Library）中导入 Image 模块，用于图像的处理
from torch.utils.tensorboard import SummaryWriter  # 从 PyTorch 的 tensorboard 模块中导入 SummaryWriter 类，用于记录数据
from torchvision import transforms  # 从 torchvision 库中导入 transforms 模块，用于图像转换

img_path = "data/train/ants_image/0013035.jpg"  # 指定图像文件的路径
img = Image.open(img_path)  # 使用 PIL 的 Image.open() 方法打开图像文件，并返回一个 PIL 图像对象

writer = SummaryWriter("logs")  # 创建一个 SummaryWriter 对象，并指定日志目录为 "logs"，用于记录 TensorBoard 数据

tensor_trans = transforms.ToTensor()  # 创建一个 ToTensor 对象，用于将 PIL 图像转换为 PyTorch 的 Tensor
tensor_img = tensor_trans(img)  # 使用 ToTensor 对象将 PIL 图像转换为 Tensor 对象

writer.add_image("Tensor_img", tensor_img)  # 将 Tensor 图像数据添加到 TensorBoard 日志中，标签为 "Tensor_img"

writer.close()  # 关闭 SummaryWriter 对象，确保所有数据都正确写入到日志文件中
```

#### 常见的Transforms

+ 关注输入输出内容，多看官方文档
+ 输入 `image.open() # 类型为PIL`
+ 输出 `ToTensor() # 类型为tensor `
+ 作用 `cv.imread() # 类型为narrays`

```python
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter(log_dir='./logs')
img = Image.open("images/pytorch.png")

# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image('ToTensor', img_tensor)

# Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image('Normalized', img_norm, 3)

# Resize
print(img.size)
trans_resized = transforms.Resize((512, 512))
# img PIL -> resize -> img_resize PIL
img_resized = trans_resized(img)
# img_resize PIL -> totensor ->img_resize tensor
img_resized = trans_totensor(img_resized)
writer.add_image('Resized', img_resized, 0)

# Compose - resize - 2
trans_resize_2 = transforms.Resize(512)
# PIL -> PIL -> tensor
trans_compose = transforms.Compose([trans_resize_2, transforms.ToTensor()])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 1)

# RandomCrop
trans_random = transforms.RandomCrop(500, 1000)
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCropHW", img_crop, i)

writer.close()

```

```python
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# 初始化 SummaryWriter 以记录图像和数据，供 TensorBoard 可视化使用
writer = SummaryWriter(log_dir='./logs')

# 使用 PIL 打开图像文件并加载到内存中
img = Image.open("images/pytorch.png")

# ToTensor
# 创建一个将 PIL 图像转换为 PyTorch 张量的转换
trans_totensor = transforms.ToTensor()

# 将 ToTensor 转换应用于图像
img_tensor = trans_totensor(img)

# 将图像张量记录到 TensorBoard，标签为 'ToTensor'
writer.add_image('ToTensor', img_tensor)

# Normalize
# 在归一化之前打印第一个通道的第一个像素值
print(img_tensor[0][0][0])

# 创建一个 Normalize 转换，指定均值和标准差
trans_norm = transforms.Normalize([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])

# 将归一化转换应用于图像张量
img_norm = trans_norm(img_tensor)

# 在归一化之后打印第一个通道的第一个像素值
print(img_norm[0][0][0])

# 将归一化后的图像张量记录到 TensorBoard，标签为 'Normalized'
writer.add_image('Normalized', img_norm, 3)

# Resize
# 打印图像的原始大小
print(img.size)

# 创建一个将图像调整为 512x512 像素大小的转换
trans_resized = transforms.Resize((512, 512))

# 将 Resize 转换应用于 PIL 图像，得到一个调整大小后的 PIL 图像
img_resized = trans_resized(img)

# 将调整大小后的 PIL 图像转换为张量
img_resized = trans_totensor(img_resized)

# 将调整大小后的图像张量记录到 TensorBoard，标签为 'Resized'
writer.add_image('Resized', img_resized, 0)

# Compose - resize - 2
# 创建一个 Resize 转换，将较小边缘调整为 512 像素，同时保持纵横比
trans_resize_2 = transforms.Resize(512)

# 组合一个先调整图像大小然后将其转换为张量的转换
trans_compose = transforms.Compose([trans_resize_2, transforms.ToTensor()])

# 将组合转换应用于 PIL 图像
img_resize_2 = trans_compose(img)

# 将调整大小后的图像张量记录到 TensorBoard，标签为 'Resize'
writer.add_image("Resize", img_resize_2, 1)

# RandomCrop
# 创建一个随机裁剪转换，将图像随机裁剪为 500x1000 像素的大小
trans_random = transforms.RandomCrop(500, 1000)

# 组合一个先应用随机裁剪然后将结果转换为张量的转换
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])

# 多次应用组合转换以获取不同的随机裁剪
for i in range(10):
    img_crop = trans_compose_2(img)
    
    # 将每个裁剪后的图像张量记录到 TensorBoard，标签为 'RandomCropHW'
    writer.add_image("RandomCropHW", img_crop, i)

# 关闭 SummaryWriter 以将所有数据刷新到磁盘
writer.close()
```

#### torchvision中数据集的使用

+ 常见数据集在**https://pytorch.org/pytorch-domains** 中

```python
import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),

])
trans_set = torchvision.datasets.CIFAR10(root="./dataset_11", train=True, transform=dataset_transforms, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset_11", train=False, transform=dataset_transforms, download=True)

writer = SummaryWriter("p10")
for i in range(10):
    img, target = trans_set[i]
    writer.add_image("test_set", img, i)

writer.close()
```

```python
import torchvision  # 导入torchvision库，用于计算机视觉相关操作
from torch.utils.tensorboard import SummaryWriter  # 从torch.utils.tensorboard导入SummaryWriter类，用于记录训练过程

# 定义数据集转换操作，将图像转换为Tensor类型
dataset_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),  # 将图像数据转换为Tensor

])

# 加载CIFAR-10训练数据集，应用上述定义的转换操作
trans_set = torchvision.datasets.CIFAR10(
    root="./dataset_11",  # 指定数据集的根目录
    train=True,  # 指定加载训练集
    transform=dataset_transforms,  # 应用转换操作
    download=True  # 如果数据集不存在则下载
)

# 加载CIFAR-10测试数据集，应用上述定义的转换操作
test_set = torchvision.datasets.CIFAR10(
    root="./dataset_11",  # 指定数据集的根目录
    train=False,  # 指定加载测试集
    transform=dataset_transforms,  # 应用转换操作
    download=True  # 如果数据集不存在则下载
)

# 创建一个SummaryWriter对象，用于写入TensorBoard日志
writer = SummaryWriter("p10")

# 循环遍历训练数据集的前10个样本
for i in range(10):
    img, target = trans_set[i]  # 获取第i个样本的图像和标签
    writer.add_image("test_set", img, i)  # 将图像写入TensorBoard，标签为“test_set”

writer.close()  # 关闭SummaryWriter对象
```

### Dataloader

+ 为后边的网络提供不同的数据形式  

```python
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10(root='./dataset_11', train=False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False, num_workers=0, drop_last=True)

img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter('dataloader')
for epoch in range(2):
    # noinspection PyRedeclaration
    step = 0
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images("Epoch: {}".format(epoch), imgs, step)
        step = step + 1

writer.close()
```

```python
# 导入 torchvision 库，它提供了用于 PyTorch 的常用数据集、模型架构和图像转换工具。
import torchvision
# 从 torch.utils.data 模块中导入 DataLoader 类，用于批量加载数据以进行训练或评估。
from torch.utils.data import DataLoader
# 从 torch.utils.tensorboard 导入 SummaryWriter 类，用于在 TensorBoard 中记录和可视化数据。
from torch.utils.tensorboard import SummaryWriter

# 加载 CIFAR-10 测试数据集。
# - `root='./dataset_11'` 指定数据存储的目录。
# - `train=False` 表示我们需要的是测试数据集，而不是训练数据集。
# - `transform=torchvision.transforms.ToTensor()` 将图像数据转换为张量格式。
test_data = torchvision.datasets.CIFAR10(root='./dataset_11', train=False, transform=torchvision.transforms.ToTensor())

# 创建一个数据加载器 (DataLoader)。
# - `dataset=test_data` 指定要加载的数据集。
# - `batch_size=64` 设置每个批次的数据量为 64。
# - `shuffle=False` 表示不打乱数据。
# - `num_workers=0` 指定用于数据加载的子进程数为 0。
# - `drop_last=True` 如果数据集不能被整除，将丢弃最后一个不完整的批次。
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False, num_workers=0, drop_last=True)

# 从测试数据集中获取第一张图像和对应的目标标签。
img, target = test_data[0]
# 打印图像的形状（通道数、高度、宽度）。
print(img.shape)
# 打印目标标签（即图像的类别标签）。
print(target)

# 创建一个 SummaryWriter 实例，用于记录数据以供 TensorBoard 可视化。
writer = SummaryWriter('dataloader')
# 进行两个 epoch 的数据记录。
for epoch in range(2):
    # 初始化步数计数器。
    step = 0
    # 遍历数据加载器中的每个批次数据。
    for data in test_loader:
        # 解包批次数据为图像和对应的目标标签。
        imgs, targets = data
        # 将批次图像添加到 TensorBoard 中，标注为当前 epoch 和步数。
        writer.add_images("Epoch: {}".format(epoch), imgs, step)
        # 增加步数计数器。
        step = step + 1

# 关闭 SummaryWriter，释放资源。
writer.close()
```

# 神经网络

## 神经网络-容器层Containers

包含六个模版

+ Module：定义了两个函数，初始化函数调用父类，forward函数处理input数据在返回数据（前向传播）

```python
import torch
from torch import nn

class Tudui(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = input + 1
        return output

tudui = Tudui()
x = torch.tensor(1.0)
output = tudui(x)
print(output)
```

```python
# 导入 PyTorch 库
import torch
# 从 torch 中导入神经网络模块
from torch import nn

# 定义一个继承自 nn.Module 的类 Tudui
class Tudui(nn.Module):
    # 初始化函数
    def __init__(self, *args, **kwargs) -> None:
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)

    # 前向传播函数
    def forward(self, input):
        # 将输入加1
        output = input + 1
        # 返回输出结果
        return output

# 创建 Tudui 类的一个实例
tudui = Tudui()
# 创建一个张量 x，其值为 1.0
x = torch.tensor(1.0)
# 将 x 传入 Tudui 类的实例中进行前向传播，得到输出
output = tudui(x)
# 打印输出结果
print(output)
```

+ Sequential
+ ModuleList
+ ModuleDict
+ ParameterList
+ ParameterDict

## 神经网络-卷积层Convolution layers

卷积层主要进行一个卷积操作-将输入图像和卷积核（权重矩阵）进行计算，相乘再相加得到一个**卷积核**。作用：提取图像特征。

+ 原理

```python
import torch
import torch.nn.functional as F

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

print(input.shape)
print(kernel.shape)

output = F.conv2d(input, kernel, stride=1)
print(output)

output2 = F.conv2d(input, kernel, stride=2)
print(output2)

output3 = F.conv2d(input, kernel, stride=1, padding=1)
print(output3)
```

```python
import torch
import torch.nn.functional as F  # 导入 PyTorch 的功能模块

# 定义一个 5x5 的输入张量
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

# 定义一个 3x3 的卷积核（kernel）
kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

# 将输入张量的形状重塑为 (1, 1, 5, 5)，
# 其中第一个维度是批量大小（batch size），第二个维度是通道数（channel）
input = torch.reshape(input, (1, 1, 5, 5))

# 将卷积核的形状重塑为 (1, 1, 3, 3)，
# 这里的维度同样表示批量大小和通道数
kernel = torch.reshape(kernel, (1, 1, 3, 3))

# 打印输入张量的形状
print(input.shape)

# 打印卷积核的形状
print(kernel.shape)

# 使用卷积核对输入张量进行卷积操作，步长为 1
output = F.conv2d(input, kernel, stride=1)
print(output)  # 输出卷积结果

# 使用卷积核对输入张量进行卷积操作，步长为 2
output2 = F.conv2d(input, kernel, stride=2)
print(output2)  # 输出卷积结果

# 使用卷积核对输入张量进行卷积操作，步长为 1，填充为 1
output3 = F.conv2d(input, kernel, stride=1, padding=1)
print(output3)  # 输出卷积结果
```

+ 应用

```python
import torch
import torchvision
from sipbuild.generator import outputs
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./data_18", train=False, transform=torchvision.transforms.ToTensor(),
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

writer = SummaryWriter("./logs_18")

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

```

```python
import torch  # 导入PyTorch库，用于深度学习
import torchvision  # 导入torchvision库，用于计算机视觉中的数据集和图像转换
from sipbuild.generator import outputs  # 从sipbuild.generator模块中导入outputs（不常见模块）
from torch import nn  # 从torch库中导入神经网络模块
from torch.utils.data import DataLoader  # 导入DataLoader模块，用于批量加载数据
from torch.utils.tensorboard import SummaryWriter  # 导入SummaryWriter，用于将信息写入TensorBoard

# 加载CIFAR-10数据集，存储在"./data_18"目录中
dataset = torchvision.datasets.CIFAR10(
    "./data_18",  # 数据集存储路径
    train=False,  # 不使用训练集，而是使用测试集
    transform=torchvision.transforms.ToTensor(),  # 将图像数据转换为Tensor
    download=True  # 如果数据集不存在，则从网上下载
)

# 创建DataLoader，进行批量数据加载，每批加载64个样本
dataloader = DataLoader(dataset, batch_size=64)

# 定义一个名为Tudui的神经网络类，继承自nn.Module
class Tudui(nn.Module):
    def __init__(self):  # 初始化方法
        super(Tudui, self).__init__()  # 调用父类的初始化方法
        # 定义一个二维卷积层，输入通道数为3，输出通道数为6，卷积核大小为3x3
        self.conv1 = nn.Conv2d(3, 6, 3, stride=1, padding=0)

    # 定义前向传播函数
    def forward(self, x):
        # 将输入数据通过卷积层，并返回结果
        x = self.conv1(x)
        return x

# 实例化Tudui类，创建一个神经网络对象
tudui = Tudui()

# 创建SummaryWriter对象，指定日志文件存储路径为"./logs_18"
writer = SummaryWriter("./logs_18")

step = 0  # 初始化步数为0
# 遍历dataloader中的数据
for data in dataloader:
    images, labels = data  # 解包数据，将图像和标签分别赋值给images和labels
    output = tudui(images)  # 将图像输入到神经网络中，得到输出
    print(images.shape)  # 打印输入图像的形状
    print(output.shape)  # 打印输出结果的形状
    # 输入图像形状：torch.Size([64, 3, 32, 32])
    
    # 将输入图像添加到TensorBoard，标签为"input"
    writer.add_images("input", images, step)
    # 输出结果形状：torch.Size([64, 6, 30, 30])

    # 将输出结果重塑为[-1, 3, 30, 30]的形状
    output = torch.reshape(output, [-1, 3, 30, 30])
    # 将输出结果添加到TensorBoard，标签为"output"
    writer.add_images("output", output, step)
    step += 1  # 步数加1
```

## 神经网络-池化层Pooling layers

池化方式很多，例如上采样，下采样......主要目的是采样最大值或者最小值或者均值（提纯）。作用：降低图像特征，保证特征的同时压缩数据量，大幅减少网络的参数量，进行加速训练。

+ kernel_size（池化核是几乘几的）。在Max Pool2d的情况下，default stride = kernel_size。
+ stride
+ padding
+ dilation(空洞卷积)
+ return_indices
+ ceil_mode（取整方式）当采样不完整的时候，是否进行采样

```python
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

writer = SummaryWriter('./logs_maxpool')
step = 0

for data in dataloader:
    imgs, targets = data
    writer.add_images('input', imgs, step)
    outputs = tudui(imgs)
    writer.add_images('outputs', outputs, step)
    step += 1

writer.close()
```

```python
import torch  # 导入PyTorch库
import torchvision  # 导入torchvision库，用于处理图像数据集
from torch import nn  # 从PyTorch中导入神经网络模块
from torch.nn import MaxPool2d  # 从神经网络模块中导入2D最大池化层
from torch.utils.data import DataLoader  # 导入DataLoader类，用于批量加载数据
from torch.utils.tensorboard import SummaryWriter  # 导入SummaryWriter，用于记录数据以便在TensorBoard中可视化

# 加载CIFAR-10数据集
dataset = torchvision.datasets.CIFAR10(
    root='./data_18',  # 数据集存储路径
    train=False,  # 指定使用测试集
    download=True,  # 如果数据集不存在则下载
    transform=torchvision.transforms.ToTensor()  # 将图像转换为Tensor格式
)

# 使用DataLoader批量加载数据，batch_size=64表示每次加载64个样本
dataloader = DataLoader(dataset, batch_size=64)

# 定义一个名为Tudui的神经网络类，继承自nn.Module
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()  # 调用父类的初始化方法
        # 定义一个2D最大池化层，池化窗口大小为3x3，使用ceil_mode=False
        self.maxpool = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        # 应用最大池化层处理输入数据
        output = self.maxpool(input)
        return output  # 返回池化后的输出

# 创建Tudui类的实例
tudui = Tudui()

# 创建SummaryWriter实例，用于记录数据以便在TensorBoard中可视化
writer = SummaryWriter('./logs_maxpool')
step = 0  # 初始化步骤计数器

# 遍历数据加载器，处理每个批次的数据
for data in dataloader:
    imgs, targets = data  # 解包数据，imgs为图像数据，targets为对应的标签
    writer.add_images('input', imgs, step)  # 将输入图像记录到TensorBoard
    outputs = tudui(imgs)  # 使用模型对输入图像进行前向传播，获取输出
    writer.add_images('outputs', outputs, step)  # 将模型输出记录到TensorBoard
    step += 1  # 增加步骤计数器

# 关闭SummaryWriter，释放资源
writer.close()
```

## 神经网络-填充层Padding layers

不常用

## 神经网络-非线性激活Non-linear Activations

将数据进行一种截断或者更改，目的：网络当中引入一种非线性特征，才能训练出符合各种特征的模型。

```python
import torch
from torch import nn

input = torch.tensor([[1, -0.5],
                      [-1, 3]])

output = torch.reshape(input, (-1, 1, 2, 2))
print(output.shape)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.relu(input)
        return output

tudui = Tudui()
output = tudui(input)
print(output)
```

```python
import torch  # 导入PyTorch库
from torch import nn  # 从PyTorch库中导入神经网络模块

# 定义一个2x2的输入张量
input = torch.tensor([[1, -0.5],
                      [-1, 3]])

# 将输入张量重塑为四维张量，形状为(batch_size, channels, height, width)
# -1表示自动计算批量大小，这里为1，即张量变为(1, 1, 2, 2)
output = torch.reshape(input, (-1, 1, 2, 2))
print(output.shape)  # 打印重塑后的张量形状，输出: torch.Size([1, 1, 2, 2])

# 定义一个名为Tudui的神经网络类，继承自nn.Module
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()  # 调用父类的初始化方法
        # 定义一个ReLU激活函数
        self.relu = nn.ReLU()

    def forward(self, input):
        # 应用ReLU激活函数，将输入张量中的负值置为0
        output = self.relu(input)
        return output  # 返回ReLU激活后的输出

# 创建Tudui类的实例
tudui = Tudui()

# 使用模型对输入数据进行前向传播，应用ReLU激活函数
output = tudui(input)
print(output)  # 打印ReLU激活后的输出
```

```python
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='./data_18', train=False, download=True,
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
writer = SummaryWriter(log_dir='./logs_20')
for data in dataloader:
    imgs, labels = data
    writer.add_images('input', imgs, step)
    outputs = tudui(imgs)
    writer.add_images('outputs', outputs, step)
    step += 1

writer.close()

```

```python
import torch  # 导入PyTorch库
import torchvision  # 导入torchvision库，用于处理图像数据集
from torch import nn  # 从PyTorch中导入神经网络模块
from torch.utils.data import DataLoader  # 导入DataLoader类，用于批量加载数据
from torch.utils.tensorboard import SummaryWriter  # 导入SummaryWriter，用于记录数据以便在TensorBoard中可视化

# 加载CIFAR-10数据集
dataset = torchvision.datasets.CIFAR10(
    root='./data_18',  # 数据集存储路径
    train=False,  # 指定使用测试集
    download=True,  # 如果数据集不存在则下载
    transform=torchvision.transforms.ToTensor()  # 将图像转换为Tensor格式
)

# 使用DataLoader批量加载数据，batch_size=64表示每次加载64个样本
dataloader = DataLoader(dataset, batch_size=64)

# 定义一个名为Tudui的神经网络类，继承自nn.Module
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()  # 调用父类的初始化方法
        # 定义一个ReLU激活函数
        self.relu = nn.ReLU()
        # 定义一个Sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # 应用Sigmoid激活函数，将输入张量中的每个元素映射到[0, 1]之间
        output = self.sigmoid(input)
        return output  # 返回Sigmoid激活后的输出

# 创建Tudui类的实例
tudui = Tudui()

# 初始化TensorBoard的SummaryWriter，指定日志存储路径为'./logs_20'
writer = SummaryWriter(log_dir='./logs_20')
step = 0  # 初始化步骤计数器

# 遍历数据加载器，处理每个批次的数据
for data in dataloader:
    imgs, labels = data  # 解包数据，imgs为图像数据，labels为对应的标签
    writer.add_images('input', imgs, step)  # 将输入图像记录到TensorBoard
    outputs = tudui(imgs)  # 使用模型对输入图像进行前向传播，获取输出
    writer.add_images('outputs', outputs, step)  # 将模型输出记录到TensorBoard
    step += 1  # 增加步骤计数器

# 关闭SummaryWriter，释放资源
writer.close()
```

## 神经网络-归一化层Normalization layers 

用于对输入数据进行标准化处理的一种层。其主要作用是对输入数据或中间特征进行缩放和偏移，以便于模型更好地训练，提高模型的训练效率和泛化能力。

## 神经网络-循环层Recurrent layers

处理序列数据的一类层。它们可以在输入数据中保持和利用过去的信息，非常适合处理时间序列数据、自然语言处理、语音识别等任务。

## 神经网络-变压器层Transformer layers

现代深度学习中非常重要的组件，尤其是在自然语言处理（NLP）领域。变压器层的设计主要依赖于注意力机制，特别是多头自注意力机制，以高效处理序列数据。

## 神经网络-线性层Linear layers

行线性变换，将输入向量映射到输出向量，通过调整权重和偏置，能够学习数据之间的线性关系

```python
import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader


dataset = torchvision.datasets.CIFAR10(root='./data_18', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output

tudui = Tudui()

for data in dataloader:
    imgs, labels = data
    print(imgs.shape)
    output = torch.flatten(imgs)
    print(output.shape)
    output = tudui(output)
    print(output.shape)
```

```python
import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

# 从torchvision中加载CIFAR10数据集，设置root参数为'./data_18'，表示数据集存储路径
# 设置train=False表示加载测试集，download=True表示如果数据集不存在则下载
# transform参数将图片转换为Tensor格式
dataset = torchvision.datasets.CIFAR10(root='./data_18', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

# 使用DataLoader来批量加载数据集，设置batch_size=64表示每批次加载64张图片
dataloader = DataLoader(dataset, batch_size=64)

# 定义一个名为Tudui的神经网络模型，继承自nn.Module
class Tudui(nn.Module):
    def __init__(self):
        # 调用父类的初始化方法
        super().__init__()
        # 定义一个线性层，将输入特征维度从196608映射到10（对应CIFAR10的10个类别）
        self.linear1 = Linear(196608, 10)

    # 定义前向传播方法
    def forward(self, input):
        # 将输入通过线性层，得到输出
        output = self.linear1(input)
        return output

# 实例化Tudui模型
tudui = Tudui()

# 遍历数据加载器，获取每批次的数据
for data in dataloader:
    imgs, labels = data  # imgs是图片张量，labels是对应的标签
    print(imgs.shape)  # 输出图片张量的形状

    # 将imgs张量展平成一维，作为线性层的输入
    output = torch.flatten(imgs)
    print(output.shape)  # 输出展平后张量的形状

    # 将展平后的张量传入模型，得到输出
    output = tudui(output)
    print(output.shape)  # 输出模型的结果形状
```

## 神经网络-丢弃层Dropout layers

在训练过程中随机丢弃一部分神经元的机制

## 神经网络-稀疏层Sparse layers

有助于提高计算效率，减少内存占用，并且在某些情况下还可以提高模型的泛化能力

## 神经网络实战

```python
import torch
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.tensorboard import SummaryWriter


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, kernel_size=5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )
    def forward(self, x):
        x = self.model1(x)
        return x

tudui = Tudui()
print(tudui)
input = torch.ones((64, 3, 32, 32))
output = tudui(input)
print(output.shape)

Writer = SummaryWriter(log_dir='../logs/logs_seq')
Writer.add_graph(tudui, input)
Writer.close()
```

```python
import torch
from torch import nn  # 导入 PyTorch 的神经网络模块
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear  # 从 nn 模块中导入常用的神经网络层
from torch.utils.tensorboard import SummaryWriter  # 导入 SummaryWriter 用于记录日志

# 定义一个自定义的神经网络类 Tudui，继承自 nn.Module
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()  # 调用父类的构造函数初始化
        self.model1 = Sequential(  # 使用 Sequential 组合多个神经网络层
            Conv2d(3, 32, kernel_size=5, padding=2),  # 2D 卷积层：输入通道数为 3，输出通道数为 32，卷积核大小为 5x5，使用 2 像素的填充
            MaxPool2d(2),  # 2D 最大池化层：池化窗口大小为 2x2
            Conv2d(32, 32, kernel_size=5, padding=2),  # 另一个卷积层：输入通道数为 32，输出通道数为 32
            MaxPool2d(2),  # 最大池化层：窗口大小为 2x2
            Conv2d(32, 64, kernel_size=5, padding=2),  # 卷积层：输入通道数为 32，输出通道数为 64
            MaxPool2d(2),  # 最大池化层：窗口大小为 2x2
            Flatten(),  # 展平层：将多维的卷积图像展平成一维向量
            Linear(1024, 64),  # 全连接层：输入特征数为 1024，输出特征数为 64
            Linear(64, 10)  # 全连接层：输入特征数为 64，输出特征数为 10
        )

    def forward(self, x):
        x = self.model1(x)  # 前向传播：将输入 x 通过模型中的层依次传递
        return x  # 返回最终输出

tudui = Tudui()  # 实例化自定义网络
print(tudui)  # 打印网络结构

input = torch.ones((64, 3, 32, 32))  # 创建一个形状为 (64, 3, 32, 32) 的全 1 张量作为输入（batch size=64，3 通道，32x32 图片）
output = tudui(input)  # 将输入数据通过网络进行前向传播
print(output.shape)  # 打印输出的形状（应为 64x10）

Writer = SummaryWriter(log_dir='../logs/logs_seq')  # 创建 SummaryWriter 实例，用于记录网络结构和数据日志
Writer.add_graph(tudui, input)  # 将网络模型和输入添加到日志中，用于在 TensorBoard 中可视化
Writer.close()  # 关闭 SummaryWriter
```

## 神经网络-损失函数Loss Functions

+ 损失函数：
  + 计算实际输出值和目标之间的差距。
  + 为我们更新输出提供一定的依据（反向传播），grad梯度。

```python
import torch
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

dataset = torchvision.datasets.CIFAR10(root='../data/data_18', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=1)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, kernel_size=5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )
    def forward(self, x):
        x = self.model1(x)
        return x

loss = nn.CrossEntropyLoss()
tudui = Tudui()
for data in dataloader:
    imgs, targets = data
    output = tudui(imgs)
    result_loss = loss(output, targets)
    result_loss.backward()
    print('ok')
```

```python
import torch  # 导入PyTorch库
import torchvision  # 导入torchvision库
from torch import nn  # 从torch库中导入神经网络模块
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear  # 从nn模块导入常用的层
from torch.utils.data import DataLoader  # 导入数据加载器
from torch.utils.tensorboard import SummaryWriter  # 导入tensorboard工具
from torchvision import transforms  # 导入数据变换工具

# 加载CIFAR10数据集，存储路径为'../data/data_18'，设置不进行训练模式，并自动下载
dataset = torchvision.datasets.CIFAR10(root='../data/data_18', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
# 创建数据加载器，batch_size设置为1
dataloader = DataLoader(dataset, batch_size=1)

# 定义一个名为Tudui的神经网络模型，继承自nn.Module
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()  # 调用父类的构造函数
        # 定义模型的第一部分model1，由多个层组成的序列
        self.model1 = Sequential(
            Conv2d(3, 32, kernel_size=5, padding=2),  # 卷积层，输入通道数3，输出通道数32，卷积核大小5x5，填充2
            MaxPool2d(2),  # 最大池化层，池化窗口大小2x2
            Conv2d(32, 32, kernel_size=5, padding=2),  # 卷积层，输入输出通道数32，卷积核大小5x5，填充2
            MaxPool2d(2),  # 最大池化层，池化窗口大小2x2
            Conv2d(32, 64, kernel_size=5, padding=2),  # 卷积层，输入通道数32，输出通道数64，卷积核大小5x5，填充2
            MaxPool2d(2),  # 最大池化层，池化窗口大小2x2
            Flatten(),  # 展平层，将多维张量展平成一维
            Linear(1024, 64),  # 全连接层，输入大小1024，输出大小64
            Linear(64, 10)  # 全连接层，输入大小64，输出大小10
        )

    def forward(self, x):  # 定义前向传播方法
        x = self.model1(x)  # 将输入x传入模型的第一部分
        return x  # 返回输出

loss = nn.CrossEntropyLoss()  # 定义交叉熵损失函数
tudui = Tudui()  # 创建Tudui模型的实例

# 遍历数据加载器，进行模型训练
for data in dataloader:
    imgs, targets = data  # 解包数据和标签
    output = tudui(imgs)  # 获取模型输出
    result_loss = loss(output, targets)  # 计算损失
    result_loss.backward()  # 反向传播计算梯度
    print('ok')  # 打印确认信息
```

## 神经网络-优化器Opimizer

调整模型参数以最小化损失函数（或代价函数），从而提高模型的性能和精确度。

```python
import torch
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

dataset = torchvision.datasets.CIFAR10(root='../data/data_18', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=1)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, kernel_size=5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )
    def forward(self, x):
        x = self.model1(x)
        return x

loss = nn.CrossEntropyLoss()
tudui = Tudui()
optim = torch.optim.SGD(tudui.parameters(), lr=0.01)
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        output = tudui(imgs)
        result_loss = loss(output, targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        running_loss = running_loss + result_loss
    print(running_loss)
```

```python
import torch  # 导入PyTorch库
import torchvision  # 导入torchvision库，用于处理视觉数据
from torch import nn  # 从torch中导入神经网络模块
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear  # 导入神经网络相关的类
from torch.utils.data import DataLoader  # 导入数据加载器类
from torch.utils.tensorboard import SummaryWriter  # 导入用于TensorBoard可视化的类
from torchvision import transforms  # 导入数据转换模块

# 下载CIFAR10数据集，并将其转换为Tensor
dataset = torchvision.datasets.CIFAR10(
    root='../data/data_18',  # 数据集存储的路径
    train=False,  # 设置为False表示加载测试集
    download=True,  # 如果数据集不存在则进行下载
    transform=torchvision.transforms.ToTensor()  # 将数据转换为Tensor
)

# 创建数据加载器，设置批量大小为1
dataloader = DataLoader(dataset, batch_size=1)

# 定义一个神经网络类Tudui，继承自nn.Module
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()  # 初始化父类
        self.model1 = Sequential(  # 使用Sequential容器按顺序定义网络层
            Conv2d(3, 32, kernel_size=5, padding=2),  # 2D卷积层，输入通道3，输出通道32，卷积核大小5，边缘填充2
            MaxPool2d(2),  # 2D最大池化层，窗口大小2
            Conv2d(32, 32, kernel_size=5, padding=2),  # 2D卷积层
            MaxPool2d(2),  # 2D最大池化层
            Conv2d(32, 64, kernel_size=5, padding=2),  # 2D卷积层
            MaxPool2d(2),  # 2D最大池化层
            Flatten(),  # 展平操作，将多维输入一维化
            Linear(1024, 64),  # 全连接层，输入1024个神经元，输出64个神经元
            Linear(64, 10)  # 全连接层，输入64个神经元，输出10个神经元（分类为10类）
        )

    # 定义前向传播方法
    def forward(self, x):
        x = self.model1(x)  # 将输入x通过网络层
        return x  # 返回输出

# 定义交叉熵损失函数
loss = nn.CrossEntropyLoss()

# 实例化网络
tudui = Tudui()

# 定义优化器，使用随机梯度下降法（SGD），学习率为0.01
optim = torch.optim.SGD(tudui.parameters(), lr=0.01)

# 训练20个epoch
for epoch in range(20):
    running_loss = 0.0  # 初始化累计损失为0
    # 遍历数据加载器中的每一个批次数据
    for data in dataloader:
        imgs, targets = data  # 获取图像及其对应的标签
        output = tudui(imgs)  # 将图像输入到网络中，获取输出
        result_loss = loss(output, targets)  # 计算损失
        optim.zero_grad()  # 清除梯度
        result_loss.backward()  # 反向传播计算梯度
        optim.step()  # 更新模型参数
        running_loss = running_loss + result_loss  # 累加损失
    print(running_loss)  # 输出当前epoch的累计损失
```

# 网络模型

## 现有网络模型的使用与修改

```python
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
```

```python
import torchvision
from torch import nn

# 导入torchvision库，主要用于计算机视觉任务。
# 导入PyTorch的nn模块，用于构建神经网络。

# train_data = torchvision.datasets.ImageNet(root='../data/data_25', train=True, download=True,
#                                           transform=torchvision.transforms.ToTensor())

# 使用VGG16模型，未使用预训练权重。
vgg16_false = torchvision.models.vgg16(pretrained=False)
# 使用VGG16模型，使用预训练权重。
vgg16_true = torchvision.models.vgg16(pretrained=True)

# 打印使用预训练权重的VGG16模型的结构。
print(vgg16_true)

# 加载CIFAR-10数据集用于训练，下载数据并将图像转换为张量。
train_data = torchvision.datasets.CIFAR10(root='../data/data_18', train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())

# vgg16_true.add_module('add_linear', nn.Linear(1000, 10))
# 在VGG16模型的分类器中添加一个线性层，用于将输出调整为10个类别。
vgg16_true.classifier.add_module('add_linear', nn.Linear(in_features=1000, out_features=10))

# 打印修改后的使用预训练权重的VGG16模型的结构。
print(vgg16_true)

# 打印未使用预训练权重的VGG16模型的结构。
print(vgg16_false)

# 将未使用预训练权重的VGG16模型的分类器的第7层替换为一个新的线性层，输出为10个类别。
vgg16_false.classifier[6] = nn.Linear(in_features=4096, out_features=10)

# 打印修改后的未使用预训练权重的VGG16模型的结构。
print(vgg16_false)
```

## 网络模型的保存和加载

网络模型的保存

```python
import torch
import torch.nn as nn
import torchvision

vgg16 = torchvision.models.vgg16(pretrained=False)
# 保存方式1,模型结构+模型参数
torch.save(vgg16, "vgg16_method1.pth")

# 保存方式2，模型参数（官方推荐）
torch.save(vgg16.state_dict(), "vgg16_method2.pth")

# 陷阱
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        return x

tudui = Tudui()
torch.save(tudui, "tudui_method1.pth")
```

```python
import torch
import torch.nn as nn
import torchvision

# 加载VGG16模型，但不使用预训练参数
vgg16 = torchvision.models.vgg16(pretrained=False)

# 保存方式1：模型结构 + 模型参数
torch.save(vgg16, "vgg16_method1.pth")  # 保存整个模型（结构+参数）到文件中

# 保存方式2：仅保存模型参数（官方推荐）
torch.save(vgg16.state_dict(), "vgg16_method2.pth")  # 仅保存模型参数到文件中

# 陷阱示例 - 自定义模型类
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # 定义一个卷积层

    def forward(self, x):
        x = self.conv1(x)  # 前向传播：输入经过卷积层
        return x

# 创建自定义模型实例
tudui = Tudui()

# 保存自定义模型的结构和参数
torch.save(tudui, "tudui_method1.pth")  # 保存整个模型（结构+参数）到文件中
```

网络模型的加载

```python
import torch
import torchvision
import torch.nn as nn
from P26_model_save import *
# 方式1->保存方式1，加载模型
model = torch.load("vgg16_method1.pth")
print(model)

# 方式2->保存方式2
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load('vgg16_method2.pth'))
print(vgg16)

# 陷阱
# class Tudui(nn.Module):
#     def __init__(self):
#         super(Tudui, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         return x

model = torch.load("tudui_method1.pth")
print(model)
```

```python
import torch  # 导入 PyTorch 库
import torchvision  # 导入 torchvision 库，用于计算机视觉相关操作
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
from P26_model_save import *  # 从 P26_model_save 模块导入所有内容

# 方式1->保存方式1，加载模型
model = torch.load("vgg16_method1.pth")  # 加载通过方式1保存的模型（包括模型结构和参数）
print(model)  # 打印加载的模型结构

# 方式2->保存方式2
vgg16 = torchvision.models.vgg16(pretrained=False)  # 初始化一个未预训练的 VGG16 模型
vgg16.load_state_dict(torch.load('vgg16_method2.pth'))  # 加载通过方式2保存的模型参数（state_dict）
print(vgg16)  # 打印加载的模型结构

# 陷阱
# class Tudui(nn.Module):
#     def __init__(self):
#         super(Tudui, self).__init__()  # 调用父类的初始化函数
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # 定义一个二维卷积层

#     def forward(self, x):
#         x = self.conv1(x)  # 执行卷积操作
#         return x  # 返回卷积结果

model = torch.load("tudui_method1.pth")  # 加载通过方式1保存的自定义模型
print(model)  # 打印加载的模型结构
```

## 完整的模型训练套路

创建dataset，dataloader加载数据，设置模型的层数，定义损失函数，定义优化器，设置网络参数，开始训练，验证模型，可视化结果

```python
import torch
from torch import nn

# 搭建神经网络
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    tudui = Tudui()
    input = torch.ones(64, 3, 32, 32)
    output = tudui(input)
    print(output.shape)


```

```python
import torch  # 导入 PyTorch 库
from torch import nn  # 从 PyTorch 导入神经网络模块

# 搭建神经网络
class Tudui(nn.Module):  # 定义一个名为 Tudui 的神经网络类，继承自 nn.Module
    def __init__(self):  # 初始化函数
        super(Tudui, self).__init__()  # 调用父类的初始化函数
        self.model = nn.Sequential(  # 使用 nn.Sequential 构建一个顺序容器
            nn.Conv2d(3, 32, 5, 1, 2),  # 定义第一个卷积层，输入通道3，输出通道32，卷积核大小5，步长1，填充2
            nn.MaxPool2d(2),  # 定义第一个最大池化层，池化窗口大小2
            nn.Conv2d(32, 32, 5, 1, 2),  # 定义第二个卷积层，输入通道32，输出通道32，卷积核大小5，步长1，填充2
            nn.MaxPool2d(2),  # 定义第二个最大池化层，池化窗口大小2
            nn.Conv2d(32, 64, 5, 1, 2),  # 定义第三个卷积层，输入通道32，输出通道64，卷积核大小5，步长1，填充2
            nn.MaxPool2d(2),  # 定义第三个最大池化层，池化窗口大小2
            nn.Flatten(),  # 将多维的特征图展平成一维向量
            nn.Linear(64*4*4, 64),  # 定义第一个全连接层，输入节点64*4*4，输出节点64
            nn.Linear(64, 10)  # 定义第二个全连接层，输入节点64，输出节点10
        )

    def forward(self, x):  # 前向传播函数
        x = self.model(x)  # 将输入数据通过模型进行前向传播
        return x  # 返回输出结果

if __name__ == '__main__':  # 主程序入口
    tudui = Tudui()  # 实例化 Tudui 神经网络
    input = torch.ones(64, 3, 32, 32)  # 创建一个形状为 (64, 3, 32, 32) 的张量，所有元素为1，模拟输入数据
    output = tudui(input)  # 将输入数据通过模型，得到输出
    print(output.shape)  # 打印输出张量的形状
```

```python
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from P27_model import *
# 准备数据集
train_data = torchvision.datasets.CIFAR10(root='../data/data_18', train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root='../data/data_18', train=False,download=True,
                                         transform=torchvision.transforms.ToTensor())

# length长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为:{}".format(train_data_size))
print("测试数据集的长度为:{}".format(test_data_size))

# 利用dataloader来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
tudui = Tudui()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的论数
epoch = 10

# 添加tensorboard
writer = SummaryWriter(log_dir='../logs/log_train')

for i in range(epoch):
    print("--------第{}轮训练开始了-----------".format(i+1))

    # 训练步骤开始
    tudui.train()
    for data in train_dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数：{},Loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar('train_loss', loss.item(), total_train_step)

    # 测试步骤开始
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))
    writer.add_scalar('test_loss', total_test_loss, total_test_step)
    writer.add_scalar('test_accuracy', total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    torch.save(tudui, "tudui_{}.pth".format(i))
    # torch.save(tudui,state_dict(), "tudui_{}.pth".format(i))
    print("模型已保存")

writer.close()
```

```python
import torch  # 导入 PyTorch 库
import torchvision  # 导入 torchvision 库，用于计算机视觉相关操作
from torch import nn  # 从 PyTorch 导入神经网络模块
from torch.utils.data import DataLoader  # 从 PyTorch 导入数据加载器模块
from torch.utils.tensorboard import SummaryWriter  # 从 PyTorch 导入 TensorBoard 的日志记录模块

from P27_model import *  # 从 P27_model 模块导入所有内容

# 准备数据集
train_data = torchvision.datasets.CIFAR10(  # 加载 CIFAR-10 训练数据集
    root='../data/data_18',  # 数据集下载到的目录
    train=True,  # 指定为训练数据集
    download=True,  # 如果数据集不存在则下载
    transform=torchvision.transforms.ToTensor()  # 将图像转换为 Tensor
)

test_data = torchvision.datasets.CIFAR10(  # 加载 CIFAR-10 测试数据集
    root='../data/data_18',  # 数据集下载到的目录
    train=False,  # 指定为测试数据集
    download=True,  # 如果数据集不存在则下载
    transform=torchvision.transforms.ToTensor()  # 将图像转换为 Tensor
)

# length长度
train_data_size = len(train_data)  # 训练数据集的样本数量
test_data_size = len(test_data)  # 测试数据集的样本数量
print("训练数据集的长度为:{}".format(train_data_size))  # 打印训练数据集的样本数量
print("测试数据集的长度为:{}".format(test_data_size))  # 打印测试数据集的样本数量

# 利用dataloader来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)  # 创建训练数据加载器，批量大小为64
test_dataloader = DataLoader(test_data, batch_size=64)  # 创建测试数据加载器，批量大小为64

# 创建网络模型
tudui = Tudui()  # 实例化自定义的 Tudui 神经网络模型

# 损失函数
loss_fn = nn.CrossEntropyLoss()  # 使用交叉熵损失函数

# 优化器
learning_rate = 0.01  # 学习率
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)  # 使用随机梯度下降优化器

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0  # 初始化训练步骤计数器
# 记录测试的次数
total_test_step = 0  # 初始化测试步骤计数器
# 训练的轮数
epoch = 10  # 设置训练的总轮数

# 添加tensorboard
writer = SummaryWriter(log_dir='../logs/log_train')  # 创建 TensorBoard 日志记录器

for i in range(epoch):  # 遍历每一轮训练
    print("--------第{}轮训练开始了-----------".format(i+1))  # 打印当前训练轮数

    # 训练步骤开始
    tudui.train()  # 设置模型为训练模式
    for data in train_dataloader:  # 遍历训练数据加载器中的每一个批次
        imgs, targets = data  # 获取批次中的图像和目标
        outputs = tudui(imgs)  # 将图像输入模型得到输出
        loss = loss_fn(outputs, targets)  # 计算输出与目标之间的损失

        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数

        total_train_step += 1  # 更新训练步骤计数器
        if total_train_step % 100 == 0:  # 每训练100步输出一次日志
            print("训练次数：{},Loss:{}".format(total_train_step, loss.item()))  # 打印当前训练步骤和损失值
            writer.add_scalar('train_loss', loss.item(), total_train_step)  # 将损失值写入 TensorBoard

    # 测试步骤开始
    tudui.eval()  # 设置模型为评估模式
    total_test_loss = 0  # 初始化测试集损失累计值
    total_accuracy = 0  # 初始化测试集准确率累计值
    with torch.no_grad():  # 禁用梯度计算
        for data in test_dataloader:  # 遍历测试数据加载器中的每一个批次
            imgs, targets = data  # 获取批次中的图像和目标
            outputs = tudui(imgs)  # 将图像输入模型得到输出
            loss = loss_fn(outputs, targets)  # 计算输出与目标之间的损失
            total_test_loss += total_test_loss + loss.item()  # 累加损失值
            accuracy = (outputs.argmax(1) == targets).sum()  # 计算预测正确的样本数
            total_accuracy += accuracy  # 累加正确预测的样本数

    print("整体测试集上的Loss: {}".format(total_test_loss))  # 打印测试集上的总损失
    print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))  # 打印测试集上的总准确率
    writer.add_scalar('test_loss', total_test_loss, total_test_step)  # 将测试集损失写入 TensorBoard
    writer.add_scalar('test_accuracy', total_accuracy/test_data_size, total_test_step)  # 将测试集准确率写入 TensorBoard
    total_test_step += 1  # 更新测试步骤计数器

    torch.save(tudui, "tudui_{}.pth".format(i))  # 保存当前训练轮次的模型
    # torch.save(tudui.state_dict(), "tudui_{}.pth".format(i))  # 保存当前训练轮次的模型参数（可选）
    print("模型已保存")  # 打印模型保存完成

writer.close()  # 关闭 TensorBoard 日志记录器
```

## 完整的模型测试模型

```python
import torch
from PIL import Image
import torchvision
from torch import nn

img_path = "../data/dog.png"
img = Image.open(img_path)
print(img)

img = img.convert('RGB')
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

img = transform(img)
print(img.shape)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

model = torch.load('tudui_9.pth')
print(model)

img = torch.reshape(img, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(img)
print(output)

print(output.argmax(1))
```

```python
import torch  # 导入 PyTorch 库
from PIL import Image  # 导入用于处理图像的库
import torchvision  # 导入用于计算机视觉任务的 PyTorch 扩展库
from torch import nn  # 导入用于构建神经网络的模块

# 图像的路径
img_path = "../data/dog.png"
img = Image.open(img_path)  # 打开图像
print(img)  # 打印图像信息

# 将图像转换为RGB格式
img = img.convert('RGB')
# 定义图像转换，包括调整大小和转换为张量
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),  # 调整图像大小为32x32
    torchvision.transforms.ToTensor()  # 将图像转换为张量
])

# 应用转换到图像
img = transform(img)
print(img.shape)  # 打印转换后图像的形状

# 定义一个名为Tudui的神经网络模型
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()  # 调用父类的构造函数
        self.model = nn.Sequential(  # 使用顺序容器定义网络结构
            nn.Conv2d(3, 32, 5, 1, 2),  # 第一个卷积层，输入通道3，输出通道32，卷积核大小5，步长1，填充2
            nn.MaxPool2d(2),  # 最大池化层，池化大小2
            nn.Conv2d(32, 32, 5, 1, 2),  # 第二个卷积层，输入通道32，输出通道32
            nn.MaxPool2d(2),  # 最大池化层
            nn.Conv2d(32, 64, 5, 1, 2),  # 第三个卷积层，输入通道32，输出通道64
            nn.MaxPool2d(2),  # 最大池化层
            nn.Flatten(),  # 展平层
            nn.Linear(64*4*4, 64),  # 全连接层，输入大小64*4*4，输出大小64
            nn.Linear(64, 10)  # 全连接层，输入大小64，输出大小10
        )

    def forward(self, x):
        x = self.model(x)  # 前向传播，通过模型进行计算
        return x

# 加载预训练模型
model = torch.load('tudui_9.pth')
print(model)  # 打印模型结构

# 重新调整图像张量的形状
img = torch.reshape(img, (1, 3, 32, 32))
model.eval()  # 设置模型为评估模式
with torch.no_grad():  # 关闭梯度计算
    output = model(img)  # 获取模型的输出
print(output)  # 打印输出

# 打印输出中概率最大的类别
print(output.argmax(1))
```

# 完结

