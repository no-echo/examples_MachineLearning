from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("../logs/logs")
image_path = "../data/data/train/ants_image/5650366_e22b7e1065.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)

writer.add_image("scr", img_array, 2, dataformats='HWC')
# y = 2
for i in range(100):
    writer.add_scalar("y = 2x", 2 * i, i)
writer.close()