import os
import torch
from PIL import Image
from torchvision import transforms
from torch import nn
from torchvision.models import resnet50
import numpy as np

def predict(input_image):
    # 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 定义图像预处理操作（与训练时相同）
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载 ResNet50 模型并修改最后的全连接层
    model = resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)
    # 加载保存的模型权重
    model.load_state_dict(torch.load(''))
    model.eval()  # 将模型设置为评估模式

    # 如果输入是文件路径，使用Image.open()打开它
    if isinstance(input_image, str):
        image = Image.open(input_image).convert('RGB')  # 确保图像为RGB格式
        print(image)
    # 如果输入是NumPy数组，使用Image.fromarray()转换它
    elif isinstance(input_image, np.ndarray):
        image = Image.fromarray(input_image.astype('uint8'), 'RGB')
        #image.show()
    else:
        raise TypeError("input_image must be a file path or a NumPy array")

    # 显示图像
    #image.show()

    # 应用预处理
    image = data_transforms(image)
    image = image.unsqueeze(0).to(device)  # 增加batch维度

    # 使用模型进行预测
    with torch.no_grad():
        output = model(image)
        predicted_score = output.item()  # 将输出转为标量

    print(f"预测的美颜分数: {predicted_score}")
    return predicted_score

if __name__ == '__main__':
    # 你可以传递文件路径或NumPy数组
    predict('/home/ncu/PycharmProjects/yolov10/save/face_1.jpg')
    # 或者
    # img_bgr = cv2.imread('/home/ncu/PycharmProjects/yolov10/save/face_1.jpg')
    # predict(img_bgr)