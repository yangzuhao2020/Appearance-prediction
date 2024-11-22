import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        # 获取文件夹内所有图片的路径
        self.image_files = [f for f in os.listdir(image_folder)]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 获取图片路径
        img_name = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(img_name)
        
        if self.transform:
            image = self.transform(image)
        
        return image

# 定义预处理变换（仅转换为Tensor，后续计算均值和标准差时无需归一化）
transform = transforms.Compose([
    transforms.ToTensor()  # 将图片转换为Tensor，范围[0, 1]
])

# 创建自定义数据集实例
dataset = CustomDataset('image_nobackground', transform=transform)

# 使用 DataLoader 加载数据集
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# 初始化均值和标准差的累加器
mean = torch.zeros(3)  # R, G, B 通道的均值
std = torch.zeros(3)   # R, G, B 通道的标准差
num_samples = 5500  # 图像总数，用于平均

# 遍历数据集，累加每个图像的均值和标准差
for images in dataloader:
    # 计算当前 batch 的均值和标准差
    mean += images.mean([0, 2, 3])  # 对每个通道计算均值
    std += images.std([0, 2, 3])    # 对每个通道计算标准差
    # num_samples += images.size(0)  # 累加样本数量

# 计算全局均值和标准差
mean /= num_samples
std /= num_samples

print('Mean:', mean)
print('Std:', std)

# tensor([0.6846, 0.5996, 0.5692])
# tensor([0.3033, 0.3169, 0.3268])