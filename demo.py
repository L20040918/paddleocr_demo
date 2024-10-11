import concurrent
import os
import re
import uuid
import shutil
import time
import threading
import subprocess
import uvicorn
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel

from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from paddleocr import PaddleOCR
from detection_model.board_model import detect_with_model_board, crop_image


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[""],
    allow_credentials=True,
    allow_methods=[""],
    allow_headers=["*"],
)

class VideoTask(BaseModel):
    video_path: str

# 截帧超时处理
def timeout_handler(process, timeout):

    def handler():
        if process.poll() is None:  # 如果进程还在运行
            process.terminate()  # 终止进程
            print("ffmpeg处理超时，已终止。")

    timer = threading.Timer(timeout, handler)  # 设置定时器
    timer.start()
    return timer  # 返回定时器实例，以便在必要时取消

# 截帧
def extract_frames_from_video(video_url, output_directory, timeout_minutes=10):
    # 检查输出目录是否存在，如果不存在则创建
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    t0 = time.time()

    # 构建FFmpeg命令
    command = [
        "ffmpeg",
        "-y",
        "-i", video_url,
        "-vf", "fps=1",
        os.path.join(output_directory, "%04d.png")
    ]
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # 设置10分钟超时
        timer = timeout_handler(process, timeout_minutes * 60)

        # 等待进程结束
        stdout, stderr = process.communicate()

        # 如果进程正常结束，取消定时器
        if timer.is_alive():
            timer.cancel()
            print("帧提取完成。")
            print(f"截帧耗时: {time.time() - t0:.3f}s")
            return True
        else:
            # 超时已经被处理，无需额外操作
            pass
    except Exception as e:
        print(f"发生错误: {e}")
        return False
    finally:
        # 确保进程被正确清理
        if process.poll() is None:
            process.kill()

# 黑板检测
def process_frames(video_url, task_id):
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        try:
            # 截帧
            video_path = "wavs/" + task_id
            data = extract_frames_from_video(video_url, video_path)
            print(data)

            file_names = os.listdir(video_path)
            if file_names:
                first_file_name = file_names[0]
                first_file_path = os.path.join(video_path, first_file_name)
                image = Image.open(first_file_path)
                width, height = image.size
                print(width)

            # 明确区分学生和教师视频路径，为每个函数和路径对提交任务
            futures = {}
            futures['board'] = executor.submit(detect_with_model_board, video_path, "pts/board.pt")
            all_results = {}  # 存储结果的字典，按任务名称索引

            # 遍历完成的Future对象，按名称获取并处理结果
            for name, future in futures.items():
                task_result = future.result()
                all_results[name] = task_result  # 将结果存入对应名称的键下
            result = all_results['board']
            return result, width
        except Exception as e:
            print(f"处理视频时发生错误: {e}")
        # finally:
        #     shutil.rmtree(video_path, ignore_errors=True)

# paddle ocr 检测
def process_frame(output_directory):
    try:
        all_result = []
        # det_model_dir：字符串类型，指定自定义的检测模型路径
        # rec_model_dir：字符串类型，指定自定义的识别模型路径
        ocr = PaddleOCR(use_angle_cls=True, lang='en')  # 移动到循环外面，避免重复初始化
        for filename in os.listdir(output_directory):
            if filename.endswith(".png") and "Board" in filename:
                img_path = os.path.join(output_directory, filename)  # 修复路径拼接问题
                # 调用OCR识别
                result = ocr.ocr(img_path, cls=True)
                if result is None:
                    print(f"图片 {img_path} 的识别结果为空，可能存在问题。")
                else:
                    for idx in range(len(result)):
                        res = result[idx]
                        if res is None:
                            print(f"警告: 结果集中的元素为None, 文件: {img_path}")
                            continue
                        for line in res:
                            text = line[1][0]
                            score = line[1][1]
                            data = {"name": img_path, "text": text, "score": score}
                            all_result.append(data)
        return all_result
    except Exception as e:
        print(f"处理出错: {e}")
        return []  # 返回空列表表示没有成功处理的数据

# 结果整合
def process_data(input_data):
    try:
        result = {}
        for item in input_data:
            name_parts = item["name"].split('/')
            if name_parts:
                sub_parts = name_parts[-1].split('_')
                if sub_parts:
                    number_name = sub_parts[0]
                else:
                    continue
            else:
                continue
            score = item["score"]
            text = item["text"]
            if number_name not in result:
                result[number_name] = [[score], [text]]
            else:
                result[number_name][0].append(score)
                result[number_name][1].append(text)
        temp_result = []
        for name, values in result.items():
            avg_score = sum(values[0]) / len(values[0])
            text_length = len(''.join(values[1]))
            temp_result.append([int(name), avg_score, text_length])
        # 在所有数据处理完之后再对temp_result进行排序
        sorted_result = sorted(temp_result, key=lambda x: x[0])
        return sorted_result
    except Exception as e:
        print(f"结果整合出错: {e}")

# 找出板书变化的值
def find_blackboard(data):
    try:
        new_result = [data[0]]
        count = 0
        for part in data[1:]:
            if part[2] - new_result[-1][2] >= 3:
                count += 1
                if count >= 3:
                    new_result.append(part)
            elif part[2] - new_result[-1][2] <= -10:
                count += 1
                if count >= 3:
                    new_result.append(part)
            else:
                count = 0
        return new_result
    except Exception as e:
        print(f"找出板书变化的值环节出现问题: {e}")

# 找出相对应的图片
def find_images(input, data, output):
    if not os.path.exists(output):
        os.makedirs(output)
    try:
        for item in data:
            name = item[0]
            for filename in os.listdir(input):
                if filename.endswith(".png") and "Board" in filename:
                    names = int(re.findall(r'\d+', filename)[0])
                    if names == name:
                        shutil.copy(os.path.join(input, filename), os.path.join(output, filename))
    except Exception as e:
        print(f"找出相对应的图片环节出现问题: {e}")

@app.post("/paddleocr/demo")
async def detect_video(input:VideoTask= Body(...)):
    try:
        task_id = str(uuid.uuid4())
        put = "wavs/" + task_id
        output = "wavs/" + task_id + "_board"
        finally_output = "wavs/" + task_id + "_finally_board"
        result_board, width = process_frames(input.video_path, task_id)
        crop_image(put, result_board, output)
        result = process_frame(output)
        results = process_data(result)
        f_result = find_blackboard(results)
        find_images(output, f_result, finally_output)
        return f_result
    except Exception as e:
        error_message = f"Failed to create task: {e}"
        print(error_message)
        raise HTTPException(status_code=500, detail=error_message)

if __name__ == "__main__":
     uvicorn.run(app, host="0.0.0.0", port=8506)