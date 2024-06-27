


import torch

from PyQt5.QtCore import  QThread, pyqtSignal



device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

from auto_avsr.demo import InferencePipeline
class RecongizeThread(QThread):
    recognize_finished = pyqtSignal(str)
    load_finished = pyqtSignal()
    
    def __init__(self, pipeline: InferencePipeline):
        super().__init__()
        self.pipeline = pipeline
        
    def process_data(self, audio, video, sample_rate):
        self.audio, self.video = self.pipeline.process_data(audio, video, sample_rate)
        self.load_finished.emit()

    def run(self):
        self.audio = self.audio.to(device)
        self.video = self.video.to(device)
        self.pipeline.modelmodule.to(device)
        transcript = self.pipeline(self.audio, self.video)
        self.recognize_finished.emit(transcript)