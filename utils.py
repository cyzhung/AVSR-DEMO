import os

import torch
import numpy as np
import cv2
import dlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from PyQt5.QtGui import QImage, QPixmap



frame_width = 320
frame_height = 240


def convert_cv_qt(cv_img):
    resized_image = cv2.resize(cv_img, (frame_width, frame_height))
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    p = QPixmap.fromImage(convert_to_Qt_format)
    return p
    
    
# predictor_path = "shape_predictor/shape_predictor_68_face_landmarks.dat"
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(predictor_path)

# def crop_lip(filePath,outputName):

#     # 加载视频
#     cap = cv2.VideoCapture(filePath)
#     frame_width = int(cap.get(3))
#     frame_height = int(cap.get(4))
#     frame_rate = int(cap.get(5))  # 获取原始视频的帧率

#     # 设置视频编码器和输出视频为MP4格式
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者使用 'XVID'
#     out = cv2.VideoWriter(outputName, fourcc, frame_rate, (frame_width, frame_height))

#     # 调整长方形大小的因子
#     width_factor = 0.4
#     height_factor = 0.4
#     face_region_resized = None
#     while cap.isOpened():
#         ret, frame = cap.read()
        
#         if not ret:
#             break
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = detector(gray)

#         for face in faces:
#             landmarks = predictor(gray, face)

#             # 确定鼻子到下巴的长方形区域
#             min_x = min([landmarks.part(n).x for n in range(10, 36)])
#             max_x = max([landmarks.part(n).x for n in range(48, 68)])
#             min_y = min([landmarks.part(n).y for n in range(32, 36)])
#             max_y = max([landmarks.part(n).y for n in range(48, 68)])

#             # 调整长方形大小
#             width_adjusted = int((max_x - min_x) * width_factor)
#             height_adjusted = int((max_y - min_y) * height_factor)

#             # 提取长方形区域
#             face_region = frame[max(0, min_y):min(frame_height, max_y + height_adjusted),
#                                 max(0, min_x):min(frame_width, max_x + width_adjusted)]

#             # 调整输出大小
#             face_region_resized = cv2.resize(face_region, (frame_width, frame_height))

#             # 写入新视频
#             out.write(face_region_resized)

#     cap.release()
#     out.release()


# def create_noise_mp4(video_path ,outputName ,noise_snr=10):
#     base_path = os.path.join(*video_path.split("/")[:-1])

#     fileName = os.path.join(video_path.split("/")[-1])

#     video = VideoFileClip(video_path)  # 讀取影片
#     audio = AudioSegment.from_file(video_path) 
#     noise = AudioSegment.from_mp3("C:/Users/cyzhung/Desktop/AVSR_DEMO/female-babble-45052.mp3")        # 讀取音樂

#     if (len(noise) < len(audio)):
#         factor = (len(audio) // len(noise))+1
#         noise = noise * factor
#     noise = noise[:len(audio)]

#     snr_gain = noise_snr - audio.dBFS

#     adjusted_noise = noise + snr_gain

    

#     noise_audio = audio.overlay(noise)
#     temp_file_path = "temp_audio.wav"
#     noise_audio.export(temp_file_path, format="wav")

#     noise_audio_clip = AudioFileClip(temp_file_path)
#     output = video.set_audio(noise_audio_clip)         # 合併影片與聲音

#     output.write_videofile(outputName, temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")
    
#     os.remove(temp_file_path)


# #create_noise_mp4("C:/Users/cyzhung/Desktop/AVSR_DEMO/50001.mp4","123.mp4")