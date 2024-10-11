import re
import os
import statistics
import time
import random
from dataclasses import dataclass

import torch
from PIL import Image
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords


@dataclass
class DetectionOptions:
    weights: str = ''
    use_pt: int = 1
    source: str = ''
    img_size: int = 640
    conf_thres: float = 0.25
    iou_thres: float = 0.45
    device: str = ''
    classes = None
    agnostic_nms: bool = False
    augment: bool = False

detection_options = DetectionOptions()

# 黑板检测
def detect_with_model_board(video_path, model_weight):
    try:
        # 线程内模型加载和初始化
        device = torch.device('cuda')
        model = attempt_load(model_weight, map_location=device)
        model.eval()

        opt = DetectionOptions(source=video_path)
        source, imgsz, trace = opt.source, opt.img_size, True

        stride = int(model.stride.max())
        imgsz = check_img_size(imgsz, s=stride)

        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # 预热
        old_img_w = old_img_h = imgsz
        old_img_b = 1

        dataset = LoadImages(source, img_size=imgsz, stride=stride)

        t0 = time.time()
        result_board = []  # 假设这里填充了检测结果

        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.float()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            if device.type != 'cpu' and (
                    old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=opt.augment)[0]

            with torch.no_grad():
                pred = model(img, augment=opt.augment)[0]

            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

            file_name = path.split('/')[-1].split('.')[0]
            match = re.search(r'\d+', file_name)
            if match:
                if '.' in match.group():
                    raise ValueError(f"Frame number appears to be a float in file name: {file_name}, expecting an integer.")
                file_name = int(match.group())
            else:
                raise ValueError(f"Unable to extract integer from file name: {file_name}")
            result_dict = {
                'id': file_name,
                'board': 0,
                'screen':0,
                'bboxes': []
            }
            for i, det in enumerate(pred):
                if len(det):
                    for *xyxy, conf, cls in reversed(det):
                        bbox = [int(coord) for coord in xyxy]
                        action = f'{names[int(cls)]}'
                        confidence = round(float(conf), 2)
                        message = {"bbox": bbox, "action": action, "confidence": confidence}
                        result_dict['bboxes'].append(message)

                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                    for c in det[:, -1].unique():
                        actionTimes = (det[:, -1] == c).sum().item()
                        action = int(c)
                        if action == 0:
                            result_dict['screen'] = actionTimes
                        elif action == 1:
                            result_dict['board'] = actionTimes
            result_board.append(result_dict)
        print(f'黑板检测耗时. ({time.time() - t0:.3f}s)')
        return result_board
    except Exception as e:
        print(f"黑板检测过程出现问题{e}")
    finally:
        del model
        torch.cuda.empty_cache()

# 裁剪图片(坐标恢复)
def crop_image(image_path, result, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 遍历文件夹中的所有文件
    for file_name in os.listdir(image_path):
        file_path = os.path.join(image_path, file_name)
        # 打开图片
        try:
            image = Image.open(file_path)
            width, height = image.size  # 获取原图片的宽度和高度
            pixels = image.load()  # 获取图片像素
        except Exception as e:
            print(f"无法打开图片 {file_path}: {e}")
            continue
        image_name = file_path.split('/')[-1]
        name = os.path.splitext(image_name)[0]

        found_match = False  # 用于标记是否找到匹配的 item

        for item in result:
            if item['id'] == int(name):
                found_match = True  # 找到匹配，标记为 True
                bbox = item['bboxes']
                for box in bbox:
                    if box['action'] in ['Board','Screen']:
                        x1,y1,x2,y2 = box['bbox'][0], box['bbox'][1], box['bbox'][2], box['bbox'][3]
                        # 根据原图片的像素调整截取坐标，如果超出范围则进行处理
                        new_x1 = x1 * (width / 640)
                        new_y1 = y1 * width / 640
                        new_x2 = x2 * (width / 640)
                        new_y2 = y2 * width / 640
                        # 基于像素和坐标截取指定区域的图片
                        try:
                            cropped_image = image.crop((new_x1, new_y1, new_x2, new_y2))
                            # 添加规定像素值的处理
                            # desired_width = width
                            # desired_height = height
                            # resized_image = cropped_image.resize((desired_width, desired_height))
                        except Exception as e:
                            print(f"截取图片时出错: {e}")
                            continue
                        # 新的图片名称：id + action
                        new_image_name = f"{item['id']}_{box['action']}.png"
                        # 处理名称已存在的情况
                        counter = 1
                        while os.path.exists(os.path.join(output_path, new_image_name)):
                            new_image_name = f"{item['id']}_{box['action']}{counter}.png"
                            counter += 1
                        # 保存截取后的图片，使用新的名称
                        try:
                            cropped_image.save(os.path.join(output_path, new_image_name))
                        except Exception as e:
                            print(f"保存图片时出错: {e}")

        if not found_match:  # 如果遍历完 result 都没找到匹配，打印提示
            print(f"未找到与 {name} 匹配的 item")