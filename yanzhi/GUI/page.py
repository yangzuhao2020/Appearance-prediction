import tkinter as tk
import os
import shutil
from tkinter import filedialog
import subprocess

# 定义全局变量来存储文件路径
file_name = None

def select_image():
    global file_name
    # 打开文件选择对话框，让用户选择一个图片文件
    file_path = filedialog.askopenfilename(
        title="选择图片", # 标题栏会显示“选择图片”。
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")] # 文件选择对话框会过滤并只显示符合这些扩展名的文件。
    ) # 函数会返回用户选择的文件的完整路径

    if file_path:
        print(f"选择了图片: {file_path}")

        # 构建目标文件路径
        file_name = os.path.basename(file_path) # 从路径中提取文件名
        target_file_path = os.path.join("image_test/original_photos", file_name) # 将文件名和文件路径

        # 复制图片复制一份到目标文件夹
        shutil.copy(file_path, target_file_path) 

    else:
        print("没有选择图片")


def image_background_disapper(file_path = "image_test/original_photos",file_name = file_name):
    target_file_path = os.path.join(file_path, file_name) 
    if not target_file_path:
        print("请先选择一张图片")
        return
    try:
        # 调用外部 Python 脚本处理背景消除
        subprocess.run(["python", "sources/change_background.py", target_file_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"处理图片时发生错误: {e}")


def image_cut(file_path = "image_test/after_remove_background", file_name = file_name):
    target_file_path = os.path.join(file_path, file_name) # 将文件名和文件路径
    if not target_file_path:
        print("请先选择一张图片")
        return
    
    try:
        subprocess.run(["python", "sources/recut.py", target_file_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"处理图片时发生错误: {e}")

def get_scores(file_path = "image_processed", file_name = file_name):
    target_file_path = os.path.join(file_path, file_name) # 将文件名和文件路径相结合
    if not target_file_path:
        print("请先选择一张图片")
        return

    try:
        subprocess.run(["python", "sources/predict.py", target_file_path], check=True)
        print("得分在终端已经显示。")

    except subprocess.CalledProcessError as e:
        print(f"处理图片时发生错误: {e}")

# 创建主窗口
root = tk.Tk()
root.title("哥哥、姐姐都好看！——颜值预测机")

# 创建按钮
button_select_image = tk.Button(root, text="请上传人脸图片", command=select_image)
button_image_background_disapper = tk.Button(root, text="去除图片背景", command=image_background_disapper)
button_image_cut = tk.Button(root, text="图片裁切", command=image_cut)
button_get_scores = tk.Button(root, text="获取图片中人物的颜值", command=get_scores)

# 布局按钮
button_select_image.pack(pady=10)
button_image_background_disapper.pack(pady=10)
button_image_cut.pack(pady=10)
button_get_scores.pack(pady=10)

# 运行主循环
root.mainloop()