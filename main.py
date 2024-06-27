import os
import sys
import time

import hydra
import torchaudio
import torchvision
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import  QTimer, QCoreApplication

from gui_rt import Ui_Dialog
from auto_avsr.demo import InferencePipeline
import torch


from threads.audioThread import AudioThread
from threads.videoThread import VideoThread, TMP_OUTPUT_VIDEO, TMP_OUTPUT_LIP, TMP_OUTPUT_RECORD
from threads.combineThread import CombineThread
from threads.recongnizeThread import RecongizeThread
from threads.loadModelThread import loadModelThread
from utils import convert_cv_qt
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class VideoPlayerApp(QMainWindow, Ui_Dialog):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.setupUi(self)
        self.video_thread = VideoThread()
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.audio_thread = AudioThread()
        self.video_thread.start()
        self.audio_thread.start()
        self.pretrain_path_dict = {"audio":"./pretrain_model/asr_trlrs3vox2_base.pth",
                                   "video":"./pretrain_model/vsr_trlrwlrs2lrs3vox2avsp_base.pth",
                                   "audiovisual":"./pretrain_model/avsr_trlrwlrs2lrs3vox2avsp_base.pth",}
        
        self.timer = QTimer(self)
        self.elapsed_time = 0
        self.timer.timeout.connect(self.update_timer)


        self.buffer = ""
        self.fileName = None
        self.audio, self.video, self.noise_audio = None, None, None

        
        # VIDEO SETUP
        
        self.ExitButton =  self.ExitButton
        self.ExitButton.clicked.connect(self.exit)



        self.ModalityCBox.currentIndexChanged.connect(self.modify_cfg)

        self.buffer_timer = QTimer(self)
        self.buffer_timer.timeout.connect(self.update_buffer)

        self.Record_Button.clicked.connect(self.record)
        self.recongnize_time = 0

    def update_timer(self):
        self.elapsed_time += 1

        mins, secs = divmod(self.elapsed_time, 60)
        time_format = '{:02d}:{:02d}'.format(mins, secs)

        # 更新 QLabel 的文本
        self.timeLabel.setText(time_format)
        # 更新 GUI 显示时间，例如更新 QLabel 显示经过的时间
    
    def update_image(self, cv_img, lip_cv_img):

        qt_img = convert_cv_qt(cv_img)
        if lip_cv_img is None:
            lip_qt_img = qt_img
        else:
            lip_qt_img = convert_cv_qt(lip_cv_img)

        self.camera_label.setPixmap(qt_img)
        self.Lip_label.setPixmap(lip_qt_img)

    def update_model(self,pipeline):
        self.recongnize_thread = RecongizeThread(pipeline)
        self.recongnize_thread.recognize_finished.connect(self.on_recognize_finished)
        self.pstate.setText("模型載入完成")
        self.repaint()
        self.audio_thread.wakeup()
        self.video_thread.wakeup()

    def modify_cfg(self):
        self.audio_thread.pause()
        self.video_thread.pause()
        
        self.cfg.data.modality = self.ModalityCBox.currentText()
        self.cfg.pretrained_model_path = self.pretrain_path_dict[self.cfg.data.modality]

        self.pstate.setText("載入模型中...")
        self.repaint()

        # 创建并启动加载模型的线程
        self.load_model_thread = loadModelThread(self.cfg)
        self.load_model_thread.load_model_signal.connect(self.update_model)
        self.load_model_thread.start()

    def exit(self):
        if hasattr(self, "audio_thread"):
            self.audio_thread.terminate()
            
        if hasattr(self, "combinerThread"):
            self.combinerThread.terminate()
            
        if hasattr(self, "video_thread"):
            self.video_thread.terminate()
            
        if hasattr(self, "recongnize_thread"):
            self.recongnize_thread.terminate()
            
        # if os.path.exists(tmp_output_video):
        #     os.remove(tmp_output_video)
        #     os.remove(tmp_output_audio)
        #     os.remove(tmp_output_record)
        #     os.remove(tmp_output_lip)

        time.sleep(1)
        QCoreApplication.quit()

    def play(self):
        if self.pstate.text() == "請選擇Modality":
            QMessageBox.warning(self, "警告", "請先選擇模態")
            return
        if self.pstate.text() == "處理影片中":
            QMessageBox.warning(self, "警告", "處理影片中，請稍後")
            return
        self.audio_thread.pause()
        self.video_thread.pause()


        self.ASRText.setText("辨識中...")
        self.repaint()
        
        self.start_recognize_thread()

        self.buffer_timer.start(100)  # 每100毫秒檢查一次buffer是否有值

    def start_recognize_thread(self):
        self.recongnize_thread.start()

    def on_recognize_finished(self, transcript):
        self.buffer = transcript
        self.audio_thread.wakeup()
        self.video_thread.wakeup()

    def update_buffer(self):
        if len(self.buffer) > 0:
            self.ASRText.setText(self.buffer.lower())
            self.buffer = ""
            self.buffer_timer.stop()  # 辨識結果已經處理，停止檢查buffer
            self.recongnize_time = 0
        else:
            self.recongnize_time += 1
        
        if self.recongnize_time > 200:
            self.ASRText.setText("")

    

        
    def record(self):
        if self.pstate.text() == "請選擇Modality":
            QMessageBox.warning(self, "警告", "請先選擇模態")
            return
        if self.video_thread.ready is False or self.audio_thread.ready is False:
            self.pstate.setText("處理影片中")
            self.repaint()

            self.audio_thread.pause()
            self.video_thread.pause()
            
            self.combinerThread = CombineThread(TMP_OUTPUT_VIDEO, tmp_output_audio, TMP_OUTPUT_RECORD)
            self.combinerThread.start()

            waveform, sample_rate = torchaudio.load(tmp_output_audio, normalize=True)
            lips, _, _ = torchvision.io.read_video(TMP_OUTPUT_LIP, pts_unit="sec")
            self.recongnize_thread.process_data(waveform, lips, sample_rate)

            self.audio_thread.pause()
            self.video_thread.pause()
            
            self.pstate.setText("影片處理完成")
            self.Record_Button.setText("Start Record")
        else:
            if self.video_thread.recording is False:
                self.elapsed_time = 0  # 重置时间
                self.timer.start(1000)  # 每1000毫秒（1秒）触发一次
                self.video_thread.start_record()
                self.audio_thread.start_record()
                self.Record_Button.setText("Recording...")
            else:
                self.pstate.setText("處理影片中")
                self.repaint()

                self.timer.stop()
                lips = self.video_thread.finish_record()
                waveform, sample_rate = self.audio_thread.finish_record()
                
                self.audio_thread.pause()
                self.video_thread.pause()
                
                self.recongnize_thread.process_data(waveform, lips, sample_rate)
                
                self.audio_thread.wakeup()
                self.video_thread.wakeup()
                self.pstate.setText("影片處理完成")
                self.play()
                self.Record_Button.setText("Start Record")

    

@hydra.main(version_base="1.3", config_path="auto_avsr/configs", config_name="config")
def main(cfg):

    app = QApplication(sys.argv)
    window = VideoPlayerApp(cfg)
    window.setWindowTitle('AVSR Demo')
    window.show()
    app.exec_()
    
if __name__ == '__main__':
    main()
