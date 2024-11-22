import cv2
from yanzhi.sources.predict import predict_single_image
from ultralytics import YOLOv10
import os
import argparse
import time
import torch

parser = argparse.ArgumentParser() # 作用在于读取命令行参数！
# 检测参数
parser.add_argument('--weights', default=r"models/train4/weights/best.pt", type=str, help='weights path') # 权重
parser.add_argument('--source', default=r"dataset/videos/video1.mp4", type=str, help='img or video(.mp4)path') # 图片或者视频的地址
parser.add_argument('--save', default=r"save", type=str, help='save img or video path') # 处理后的图片保存路径
parser.add_argument('--vis', default=True, action='store_true', help='visualize image') # 可视化结果
parser.add_argument('--conf_thre', type=float, default=0.5, help='conf_thre') # 置信度
parser.add_argument('--iou_thre', type=float, default=0.5, help='iou_thre') 
# 交并比指的是交集和并集的比值，候选框与原标记框之间的比例，表示衡量定位的精准度 
opt = parser.parse_args() # 从命令行中解析用户输入的参数
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color
# 作用很简单，用于获得不同的颜色。


class Detector(object):
    def __init__(self, weight_path, conf_threshold=0.5, iou_threshold=0.5):
        self.device = device
        self.model = YOLOv10(weight_path) 
        self.conf_threshold = conf_threshold # 置信度
        self.iou_threshold = iou_threshold # 交并比
        self.names = self.model.names # 获取模型中定义的类别名称列表。

    def detect_image(self, img_bgr): # 表示输入一张rgb的图片
        results = self.model(img_bgr, verbose=True, conf=self.conf_threshold,
                             iou=self.iou_threshold, device=self.device)
        # 调用 YOLOv10 模型对输入图像 img_bgr 进行目标检测。
        bboxes_cls = results[0].boxes.cls
        # 提取每个检测框对应的类别标签。
        bboxes_conf = results[0].boxes.conf
        # 提取每个检测框的置信度分数。
        bboxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype('uint32')
        # 提取每个检测框的坐标，格式为 [x_min, y_min, x_max, y_max] 将将检测结果转换为 NumPy 数组。

        for idx in range(len(bboxes_cls)):
            box_cls = int(bboxes_cls[idx]) # 获取当前检测框的类别索引，并将其转换为整数。
            bbox_xyxy = bboxes_xyxy[idx] 
            bbox_label = self.names[box_cls] # 框选到的方框中的内容标签
            box_conf = f"{bboxes_conf[idx]:.2f}" 
            print("置信度：",box_conf) 

            if float(box_conf)>0.7:
                xmax, ymax, xmin, ymin = bbox_xyxy[2], bbox_xyxy[3], bbox_xyxy[0], bbox_xyxy[1]
                # print(xmax, ymax, xmin, ymin)
                # 打印成裁剪框的坐标。
                crop_img = img_bgr[ymin-10:ymax+10, xmin-10:xmax+10] 
                # 从原始图像 img_bgr 中裁剪出当前检测框的内容，并增加上下左右各 10 像素的边距。 
                if not os.path.exists(opt.save):
                    os.makedirs(opt.save)
                
                crop_img_path = os.path.join(opt.save, f"{bbox_label}_{idx + 1}.jpg")
                print("裁剪后的人脸保存的路径：",crop_img_path) 
                cv2.imwrite(crop_img_path, crop_img) # 将裁剪后的图像保存到指定路径。
                crop_img = cv2.imread(crop_img_path) # 读取图片

                if os.path.exists(crop_img_path):
                    print(f"File saved successfully: {crop_img_path}")
                    scores = predict_single_image(crop_img_path)
                else:
                    print(f"Error saving file: {crop_img_path}")
                    scores = None
                    scores = predict_single_image(crop_img_path)
                    
                img_bgr = cv2.rectangle(img_bgr, (xmin, ymin), (xmax, ymax), get_color(box_cls + 3), 2)
                # 在原来图像上绘制一个矩形框，并为每个框设置不同的颜色。
                
                cv2.putText(img_bgr, 
                            f'{str(bbox_label)}/{str(scores)}', # 图像上添加文本标签 
                            (xmin, ymin - 10), # 标签卫浴图像的左上角！！
                            cv2.FONT_HERSHEY_SIMPLEX, 2, # 字体类型和字体大小！ 
                            get_color(box_cls + 3), 2 # 文本的颜色，由 get_color 函数生成。
                            ) # 给检测框添加表情 和 颜值得分
                
        return img_bgr

# Example usage
if __name__ == '__main__':
    model = Detector(weight_path=opt.weights, 
                     conf_threshold=opt.conf_thre, 
                     iou_threshold=opt.iou_thre)
    images_format = ['.png', '.jpg', '.jpeg', '.JPG', '.PNG', '.JPEG']
    video_format = ['mov', 'MOV', 'mp4', 'MP4']

    if os.path.join(opt.source).split(".")[-1] not in video_format: # 判断是否属于视频格式。
        print("注意：图片处理开始")
        image_names = [name for name in os.listdir(opt.source) for item in images_format if
                       os.path.splitext(name)[1] == item]
        # 将符合格式的图片名称存放在 image_names列表中。
        for img_name in image_names:
            img_path = os.path.join(opt.source, img_name)
            img_ori = cv2.imread(img_path)
            img_vis = model.detect_image(img_ori) # 检测图片，将检测的图片框选返回。
            img_vis = cv2.resize(img_vis, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(opt.save, img_name), img_vis) 

            # if opt.vis: # 是否可视化结果
            #     cv2.imshow(img_name, img_vis)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()

    else:
        print("注意：处理视频开始。")
        capture = cv2.VideoCapture(opt.source) # 创建视频对象
        fps = capture.get(cv2.CAP_PROP_FPS) # 获取视频的帧率
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))) # 视频中图片的宽度和高度
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        # 一个视频编码器（FourCC），用于在创建视频写入对象时指定视频的编解码器
        outVideo = cv2.VideoWriter(os.path.join(opt.save, os.path.basename(opt.source).split('.')[-2] + "_out.mp4"),
                                   fourcc, # 编码器
                                   fps, # 帧率
                                   size # 视频尺寸
                                   ) # 用于将视频帧写入文件
        while True:
            ret, frame = capture.read() # ret 表示布尔值是否获得帧，frame 获得的帧。
            if not ret:
                break
            start_frame_time = time.perf_counter()
            img_vis = model.detect_image(frame)
            # 结束计时
            end_frame_time = time.perf_counter()  # 使用perf_counter进行时间记录
            # 计算每帧处理的FPS
            elapsed_time = end_frame_time - start_frame_time # 处理时间
            if elapsed_time == 0:
                fps_estimation = 0.0
            else:
                fps_estimation = 1 / elapsed_time # 每秒可以处理的帧

            h, w, c = img_vis.shape # 获取图片的维度信息
            cv2.putText(img_vis, 
                        f"FPS: {fps_estimation:.2f}", # 处理速度
                        (10, 35),  # 文字位置
                        cv2.FONT_HERSHEY_SIMPLEX, # 文字字体 
                        1.3, # 缩放比例
                        (0, 0, 255), # 文字颜色
                        2 # 线条粗细
                        )
            outVideo.write(img_vis) # 如写新的帧率
            # cv2.imshow('detect', img_vis) # 显示
            # cv2.waitKey(1)

        capture.release()
        outVideo.release()