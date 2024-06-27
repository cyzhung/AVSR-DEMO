import time

import torch
import dlib
import cv2
import numpy as np
from PyQt5.QtCore import  QThread, pyqtSignal

predictor_path = "shape_predictor/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)



TMP_DIR = "./tmp/"

TMP_OUTPUT_VIDEO = TMP_DIR + "output.mp4"
TMP_OUTPUT_LIP = TMP_DIR + "lip.mp4"
TMP_OUTPUT_RECORD = TMP_DIR + "record.mp4"

     
    
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(object, object)

    def __init__(self):
        
        super(VideoThread, self).__init__()
        self.video_frames = []
        self.lip_frames = []
        self.running = True
        self.recording = False
        self.lip = True
        self.paused = False
        self.ready = True

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("錯誤：無法訪問攝像頭。")
            self.ready = False
            self.change_pixmap_signal.emit(None, None)  # 可以發射一個信號表明攝像頭無法訪問
            return  # 結束線程或做其他錯誤處理
    def pause(self):
        self.paused = True

    def wakeup(self):
        self.paused = False

    def getLipFrame(self,frame):
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        width_factor = 1
        height_factor = 0.5
        face_region_resized = None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)

            # 确定鼻子到下巴的长方形区域
            min_x = landmarks.part(4).x
            max_x = landmarks.part(55).x
            min_y = landmarks.part(34).y
            max_y = landmarks.part(58).y

            # 调整长方形大小
            width_adjusted = int((max_x - min_x) * width_factor)
            height_adjusted = int((max_y - min_y) * height_factor)

            # 提取长方形区域
            face_region = frame[max(0, min_y):min(frame_height, max_y + height_adjusted),
                                max(0, min_x):min(frame_width, max_x + width_adjusted)]

            # 调整输出大小
            face_region_resized = cv2.resize(face_region, (frame_width, frame_height))

        return face_region_resized


    def run(self):
        prev_lip_frame = None
        while self.running:
            if not self.paused:
                ret, cv_img = self.cap.read()
                if ret:
                    if self.lip:
                        try:
                            lip_frame = self.getLipFrame(cv_img)
                            if lip_frame is None:
                                lip_frame = prev_lip_frame
                            else:
                                prev_lip_frame = lip_frame
                        except KeyboardInterrupt:
                            exit()
                        except Exception as e:    
                            lip_frame = prev_lip_frame
                    if self.recording:
                        self.video_frames.append(cv_img)
                        if self.lip:
                            self.lip_frames.append(lip_frame)

                    self.change_pixmap_signal.emit(cv_img, lip_frame)
            else:
                time.sleep(0.5)

    def start_record(self):
        self.recording = True
        
    def finish_record(self):
        self.recording = False
        self.save_video()
        
        lip_frames = np.stack(self.lip_frames, axis=0)
        self.video_frames = []
        self.lip_frames = []

        return torch.from_numpy(lip_frames)

    def save_video(self):
        # Define the codec and create VideoWriter object
        if len(self.video_frames)==0:
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
        out = cv2.VideoWriter(TMP_OUTPUT_VIDEO, fourcc, 23, (640, 480))
        lip = cv2.VideoWriter(TMP_OUTPUT_LIP, fourcc, 23, (640, 480))

        for frame in self.lip_frames:
            lip.write(frame)
        lip.release()

        for frame in self.video_frames:
            out.write(frame)
        out.release()

    