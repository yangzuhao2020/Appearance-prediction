import gradio as gr
import os
import shutil
import subprocess


# 定义文件存储路径
ORIGINAL_DIR = "image_test/original_photos"  # 原图片复制到文件的路径
BACKGROUND_REMOVED_DIR = "image_test/after_remove_background"  # 移除背景后图片保存路径
PROCESSED_DIR = "image_processed"  # 裁减人脸后图片保存的路径


def ensure_directory_exists(directory):
    """确保目录存在"""
    if not os.path.exists(directory):
        os.makedirs(directory)


# 检查上述三个目录是否存在
ensure_directory_exists(ORIGINAL_DIR)
ensure_directory_exists(BACKGROUND_REMOVED_DIR)
ensure_directory_exists(PROCESSED_DIR)


def process_image(file_path):
    """处理图片：选择、去背景、裁切和评分"""
    if not file_path:
        return [
            "未选择任何图片，请上传图片后重试。",
            None,  # 原始图片未显示
            "去除背景失败：未选择有效图片。",
            None,  # 去背景图片未显示
            "裁切图片失败：未选择有效图片。",
            None,  # 裁切图片未显示
            "评分失败：未选择有效图片。",
        ]

    try:
        file_name = os.path.basename(file_path)  # 获取文件名称
        original_file_path = os.path.join(ORIGINAL_DIR, file_name)  # 路径拼接
        shutil.copy(file_path, original_file_path)  # 图片复制到新的文件夹中

        # Step 2: 去除背景
        subprocess.run(
            ["python", "sources/change_background.py", original_file_path],
            check=True,
        )
        background_removed_path = os.path.join(BACKGROUND_REMOVED_DIR, file_name)

        # Step 3: 裁切图片
        subprocess.run(
            ["python", "sources/recut.py", background_removed_path],
            check=True,
        )
        cropped_image_path = os.path.join(PROCESSED_DIR, file_name)

        # Step 4: 获取评分
        result = subprocess.run(
            ["python", "sources/predict.py", cropped_image_path],
            capture_output=True,  # 捕获输出
            text=True,  # 输出为文本
            check=True,
        )
        score = result.stdout.strip()  # 获取评分

        return [
            f"图片上传成功: {original_file_path}",
            original_file_path,
            "背景已成功去除",
            background_removed_path,
            "图片已成功裁切",
            cropped_image_path,
            f"颜值得分: {score}",
        ]

    except FileNotFoundError as e:
        return [
            "文件路径无效，请检查图片是否存在。",
            None,
            f"去除背景失败: 文件路径无效 ({e})",
            None,
            "裁切图片失败: 文件路径无效。",
            None,
            "评分失败: 文件路径无效。",
        ]

    except subprocess.CalledProcessError as e:
        error_message = e.stderr.strip() if e.stderr else str(e)
        return [
            "图片上传成功，但后续处理失败。",
            original_file_path if "original_file_path" in locals() else None,
            f"去除背景失败: {error_message}",
            background_removed_path if "background_removed_path" in locals() else None,
            f"裁切图片失败: {error_message}",
            cropped_image_path if "cropped_image_path" in locals() else None,
            f"评分失败: {error_message}",
        ]


# 创建 Gradio 接口
iface = gr.Interface(
    fn=process_image,
    inputs=gr.File(label="请上传人脸图片", file_types=["image"]),
    outputs=[
        gr.Textbox(label="上传结果"),
        gr.Image(label="原始图片", type="filepath"),
        gr.Textbox(label="去除背景结果"),
        gr.Image(label="去除背景的图片", type="filepath"),
        gr.Textbox(label="裁切结果"),
        gr.Image(label="裁切后的图片", type="filepath"),
        gr.Textbox(label="评分结果"),
    ],
    title="哥哥、姐姐都好看！——颜值预测机",
    description="**上传一张人脸图片，我们将为您预测颜值。**",
)

# 启动 Gradio 应用
iface.launch()