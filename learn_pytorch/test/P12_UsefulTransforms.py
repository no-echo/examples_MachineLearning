from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter(log_dir='../logs/logs')
img = Image.open("../data/images/pytorch.png")

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
