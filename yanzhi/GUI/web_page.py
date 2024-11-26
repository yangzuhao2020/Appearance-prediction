import gradio as gr
import os
import shutil
import subprocess


# 定义文件存储路径
ORIGINAL_DIR = "yanzhi/image_test/original_photos"  # 原图片复制到文件的路径
BACKGROUND_REMOVED_DIR = "yanzhi/image_test/after_remove_background"  # 移除背景后图片保存路径
PROCESSED_DIR = "yanzhi/image_processed"  # 裁减人脸后图片保存的路径
ORIGINAL_VIDEO_DIR = "dataset/videos"
PROCESSED_VIDEO_DIR = "save"


def ensure_directory_exists(directory):
    """确保目录存在"""
    if not os.path.exists(directory):
        os.makedirs(directory)


# 检查上述目录是否存在
ensure_directory_exists(ORIGINAL_DIR)
ensure_directory_exists(BACKGROUND_REMOVED_DIR)
ensure_directory_exists(PROCESSED_DIR)
ensure_directory_exists(ORIGINAL_VIDEO_DIR)
ensure_directory_exists(PROCESSED_VIDEO_DIR)


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
            ["python", "yanzhi/sources/change_background.py", original_file_path],
            check=True,
        )
        background_removed_path = os.path.join(BACKGROUND_REMOVED_DIR, file_name)

        # Step 3: 裁切图片
        subprocess.run(
            ["python", "yanzhi/sources/recut.py", background_removed_path],
            check=True,
        )
        cropped_image_path = os.path.join(PROCESSED_DIR, file_name)

        # Step 4: 获取评分
        result = subprocess.run(
            ["python", "yanzhi/sources/predict.py", cropped_image_path],
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
            f"{score}",
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

def process_video(file_path):
    """处理视频：上传并调用外部脚本进行处理"""
    if not file_path:
        return ["未选择任何视频，请上传视频后重试。", None]

    try:

        file_name = os.path.basename(file_path)  # 获取文件名称
        print(file_name)
        original_file_path = os.path.join(ORIGINAL_VIDEO_DIR, file_name)  # 路径拼接
        shutil.copy(file_path, original_file_path)  # 视频复制到新的文件夹中
        processed_file_path = os.path.join(PROCESSED_VIDEO_DIR, file_name.split('.')[0] + "_out.mp4")
        subprocess.run(["python", "predict.py", "--source", file_path])

        # 检查处理后的文件是否生成
        if not os.path.exists(processed_file_path):
            return ["处理后的视频文件未生成，请检查外部脚本是否正确执行。", None]

        return [f"视频上传成功: {original_file_path}", f"视频处理成功: {processed_file_path}"]

    except Exception as e:
        return [f"视频处理失败: {str(e)}", None]


# 创建 Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("## 哥哥、姐姐都好看！—— 颜值预测机")
    gr.Markdown("### 上传图片或视频，预测颜值或处理视频内容。")

    with gr.Tab("图片处理"):
        with gr.Row():
            image_input = gr.File(label="请上传人脸图片", file_types=["image"])
        # 将每个组件直接定义为单独的输出
        upload_result = gr.Textbox(label="上传结果")
        original_image = gr.Image(label="原始图片", type="filepath")
        background_removal_result = gr.Textbox(label="去除背景结果")
        background_removed_image = gr.Image(label="去除背景的图片", type="filepath")
        cropping_result = gr.Textbox(label="裁切结果")
        cropped_image = gr.Image(label="裁切后的图片", type="filepath")
        scoring_result = gr.Textbox(label="评分结果")

        process_image_button = gr.Button("开始处理")
        process_image_button.click(
            process_image, 
            inputs=image_input, 
            outputs=[
                upload_result, 
                original_image, 
                background_removal_result, 
                background_removed_image, 
                cropping_result, 
                cropped_image, 
                scoring_result
            ]
        )

    with gr.Tab("视频处理"):
        with gr.Row():
            video_input = gr.File(label="请上传视频文件", file_types=["video"])
        upload_result_video = gr.Textbox(label="上传结果")
        processed_video = gr.Textbox(label="处理后的视频保存的位置")
        process_video_button = gr.Button("开始处理")
        process_video_button.click(
            process_video, 
            inputs=video_input, 
            outputs=[upload_result_video, processed_video]
        )


# 启动 Gradio 应用
demo.launch()

