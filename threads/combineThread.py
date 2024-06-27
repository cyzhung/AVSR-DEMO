import subprocess

from PyQt5.QtCore import  QThread, pyqtSignal


class CombineThread(QThread):
    done = pyqtSignal()
    def __init__(self, video_file, audio_file, output_file):
        super(CombineThread, self).__init__()
        self.video_file = video_file
        self.audio_file = audio_file
        self.output_file = output_file

    def combine_audio_video(self, audio_file, video_file, output_file):
        command = [
            'ffmpeg',
            '-y',
            '-i', video_file,
            '-i', audio_file,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-strict', 'experimental',
            output_file
        ]
        subprocess.run(command)

    def run(self):
        self.combine_audio_video(self.audio_file, self.video_file, self.output_file)
        self.done.emit()