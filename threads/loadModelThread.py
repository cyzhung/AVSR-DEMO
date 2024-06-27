
from PyQt5.QtCore import  QThread, pyqtSignal

from auto_avsr.demo import InferencePipeline

class loadModelThread(QThread):
    load_model_signal = pyqtSignal(object)
    def __init__(self, cfg, parent=None):
        super(loadModelThread, self).__init__(parent)
        self.cfg = cfg

    def run(self):
        self.pipeline = InferencePipeline(self.cfg)
        self.load_model_signal.emit(self.pipeline)
    def stop(self):
        self.terminate()
