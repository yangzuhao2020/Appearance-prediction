import pandas as pd
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
# 将项目根目录添加到 sys.path
sys.path.append(project_root)

from yanzhi.sources.main import EfficientNetB0, FaceBeautyDataset  # EfficientNetB0 在 main.py 文件中定义

from PIL import Image

# 参数设置
batch_size = 4
model_path = 'yanzhi/models/model_efficient.pth'  # 模型文件路径，根据需要修改
image_dir = 'yanzhi/image_processed'  # 图像文件夹路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

# 加载标签数据和数据集
dataset = FaceBeautyDataset(img_dir = image_dir, transform=transform) 
pred_loader = DataLoader(dataset, batch_size=batch_size) 

# 加载模型
model = EfficientNetB0(num_classes=1)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True)) 
model = model.to(device) 
model.eval() 

results = [] # 初始化一个空列表 predictions，用于存储模型的预测结果。

def scores(num):
    score = 65 + (num - 1.02)/(4.75 - 1.02)*35 
    if score < 95: 
        return score.item() # 返回标量
    elif score > 110:
        score = score - 20
        return score.item()
    else:
        score2 = score * 0.9
        if score2 < 95:
            return score2.item() 
        else:
            score2 * 0.9
            return score2.item() 
    
def predict(pred_loader = pred_loader):
    with torch.no_grad():
        for inputs, labels, img_names in pred_loader:
            inputs = inputs.to(device)
            outputs = model(inputs).cpu()
            outputs_np = outputs.numpy()
            # 将每个输出结果和对应的图片名称配对
            for output, img_name in zip(outputs_np, img_names):
                results.append((img_name, scores(output[0])))  # 假设模型输出是一个标量值

    # 将结果保存到 DataFrame
    results_df = pd.DataFrame(results, columns=['Image_Name', 'Prediction'])

    # 保存到 CSV 文件
    results_df.to_csv('predictions.csv', index=False)
    
    print("预测结果已保存到 predictions.csv")

# predict(pred_loader) 

def predict_single_image(image_path, transform=transform):
    # 加载并预处理单张图片
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # 添加批次维度
    image_tensor = image_tensor.to(device) 

    # 模型预测
    with torch.no_grad():
        output = model(image_tensor).cpu() 
        output_np = output.numpy()

    # 结果处理
    prediction_score = scores(output_np[0])
    prediction_score = float(f"{prediction_score:.2f}")
    return prediction_score

if __name__ == "__main__":
    
    # single_image_path = 'image_processed/H3M.jpg'
    single_image_path = sys.argv[1]
    prediction = predict_single_image(single_image_path, transform)
    print(f"预测分数: {prediction:.2f}")