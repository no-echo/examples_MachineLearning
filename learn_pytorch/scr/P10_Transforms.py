from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

''' python的用法 -> tensor数据类型
    通过ttransforms.ToTensor去解决两个问题
    1. transforms该如何使用
    2. Tensor数据类型相较于其他的区别，我们为什么需要tensor的数据类型
'''

img_path = "../data/data/train/ants_image/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("../logs/logs")

# 创建自己的工具
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("Tensor_img", tensor_img)

writer.close()