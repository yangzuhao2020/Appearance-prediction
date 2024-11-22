import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.data
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Squeeze-and-Excitation模块
class SqueezeAndExcitation(nn.Module):
    def __init__(self, in_channels, 
                 squeeze_factor: int = 4): # 缩小因子 【这个可以修改】
        super(SqueezeAndExcitation, self).__init__()
        middle_channels = in_channels // squeeze_factor # 中间通道数
        self.fc1 = nn.Linear(in_channels, middle_channels, bias=False) # 减少通道数
        self.fc2 = nn.Linear(middle_channels, in_channels, bias=False) # 增加通道数，使得可以返回相同的通道数
        self.silu = nn.SiLU(inplace=True) # 位置：Swish激活函数，被应用在第一个全连接层
        self.sigmoid = nn.Sigmoid() # 应用在第二个全连接层之后
 
    def forward(self, x):
        b, c, w, h = x.size()
        y = F.adaptive_avg_pool2d(x, (1, 1)).view(b, c) # 全局平均池化
        y = self.fc1(y) # 放入第一个全连接层，减少通道
        y = self.silu(y) 
        y = self.fc2(y) # 放入第二个全连接层，增加通道数
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y # 将权重和矩阵相乘
    
#  MBconv 
class MBconv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=6):
        super(MBconv, self).__init__()
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels * expansion),
            nn.SiLU(inplace=True)
        )
        # 使用 1 *1 的卷积扩大通道数，扩大6倍。

        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels * expansion, in_channels * expansion, kernel_size=3, stride=stride, 
                      padding=1, groups=in_channels * expansion, bias=False),
            nn.BatchNorm2d(in_channels * expansion),
            nn.SiLU(inplace=True)
        )
        # group 表示表示每个输入通道只与一个输出通道进行卷积
        
        self.se = SqueezeAndExcitation(in_channels * expansion)
        
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels * expansion, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        # 点卷积 控制通道数
        
        self.skip = stride == 1 and in_channels == out_channels

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        identity = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.se(x)
        x = self.pointwise(x)
        if self.skip:
            x += identity

        return x
 
# EfficientNet-B0
class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=1):
        super(EfficientNetB0, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False), # 将图片的尺寸变为 112*112
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        self.blocks = nn.Sequential(
            MBconv(32, 16, stride=1, expansion=1),
            MBconv(16, 24, stride=2, expansion=6),
            MBconv(24, 40, stride=2, expansion=6),
            MBconv(40, 80, stride=2, expansion=6),
            MBconv(80, 112, stride=1, expansion=6),
            MBconv(112, 192, stride=2, expansion=6),
            MBconv(192, 320, stride=1, expansion=6)
        )
        
        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        # [batch_size, 1280, 1, 1]

        self.fc = nn.Linear(1280, num_classes)    
 
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
 
# 创建数据集类型
class FaceBeautyDataset(Dataset):
    def __init__(self, labels_df = None, img_dir = None, transform=None):
        self.labels_df = labels_df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        if self.labels_df is not None and not self.labels_df.empty:
            return len(self.labels_df)
        else:
            return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        if self.labels_df is not None and not self.labels_df.empty:
            img_name = os.path.join(self.img_dir, self.labels_df.iloc[idx]['Filename']) # 图片名称
            label = self.labels_df.iloc[idx]['score'] # 获取得分
        else:
            img_name = os.path.join(self.img_dir, os.listdir(self.img_dir)[idx]) # 图片名称
            label = 0 # 设置为0

        image = Image.open(img_name) # 将图片读取到内存中

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32), img_name
    

# 创建数据加载器
def create_datasets_and_dataloaders(labels_df, image_dir, transform, batch_size):
    dataset = FaceBeautyDataset(labels_df, image_dir, transform=transform) 
    train_size = int(0.8 * len(dataset)) 
    test_size = len(dataset) - train_size 
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size]) 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4) 
    return train_loader, test_loader

# 模型的构建
def create_model(num_classes=1):
    model = EfficientNetB0(num_classes)
    return model

def train_model(model, train_loader, learning_rate, num_epochs, device):
    # 损失函数 和 优化器的选择
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

    # 优化器学习率调整策略
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # 模型的训练
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device) 

            optimizer.zero_grad() # 清除梯度
            output = model(inputs) # 前向传播
            labels = labels.unsqueeze(1) # 将标签形状从 [batch_size] 调整为 [batch_size, 1]
            loss = criterion(output, labels) # 计算损失
            loss.backward() # 反向传播
            optimizer.step() # 应用梯度
            running_loss += loss.item() 

    # 更新学习率
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}')

    model_path = '../models/model_efficient.pth'
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

def test_model(model, test_loader, device):
    model.eval()
    test_loss = 0.0
    mse_loss = nn.MSELoss()

    with torch.no_grad():
        for inputs, labels, _ in test_loader:
            inputs, labels = inputs.to(device), labels.to(device) 
            outputs = model(inputs)
            labels = labels.unsqueeze(1)
            loss = mse_loss(outputs, labels) 
            test_loss += loss.item()
        
        avg_test_loss = test_loss / len(test_loader)
        print(f'Test Loss: {avg_test_loss}')

def main():
    # 超参数的设置
    batch_size = 64
    learning_rate = 0.01
    num_epochs = 200 
    labels_df = pd.read_csv('name_to_label.csv') 
    image_dir = '../image_nobackground/' 

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # 将像素调整为 224* 224
        transforms.RandomHorizontalFlip(p=0.5),  # 数据增强：随机水平翻转
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 随机调整亮度、对比度、饱和度和色调
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.2258, 0.1792, 0.1564], std=[0.2549, 0.2048, 0.1797])
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])

    # 创建数据加载器
    train_loader, test_loader = create_datasets_and_dataloaders(labels_df, image_dir, transform, batch_size)

    # 模型构建
    model = create_model(num_classes=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 模型训练
    train_model(model, train_loader, learning_rate, num_epochs, device)

    # 模型测试
    test_model(model, test_loader, device)

if __name__ == "__main__":
    main()