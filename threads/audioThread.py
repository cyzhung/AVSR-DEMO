import time

import pyaudio
import wave
import torchaudio
from PyQt5.QtCore import  QThread

tmp_dir = "./tmp/"
tmp_output_audio = tmp_dir + "output.wav"


class AudioThread(QThread):
    def __init__(self):
        super().__init__()
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024
        self.frames = []
        self.running = True
        self.pyaudio_instance = pyaudio.PyAudio()
        self.recording = False
        try:
            self.stream = self.pyaudio_instance.open(format=self.format,
                                                    channels=self.channels,
                                                    rate=self.rate,
                                                    input=True,
                                                    frames_per_buffer=self.chunk)
            self.ready = True
        except Exception as e:
            print(f"錯誤：無法訪問麥克風。{e}")
            self.stream = None
            self.ready = False
        self.paused = False
    def pause(self):
        self.paused = True

    def wakeup(self):
        self.paused = False

    def start_record(self):
        self.recording = True

    def finish_record(self):
        self.recording = False
        return self.save_audio()
            
    def run(self):
        
        while self.running:
            if not self.paused:
                if self.recording:
                    data = self.stream.read(self.chunk, exception_on_overflow=False)
                    self.frames.append(data)
            else:
                time.sleep(0.5)

    def stop(self):
        self.running = False
        self.stream.stop_stream()
        self.stream.close()
        self.pyaudio_instance.terminate()
        self.terminate()

    def save_audio(self):
        # Define the output WAV file parameters
        wav_file = wave.open(tmp_output_audio, 'wb')
        wav_file.setnchannels(self.channels)
        wav_file.setsampwidth(self.pyaudio_instance.get_sample_size(self.format))
        wav_file.setframerate(self.rate)
        wav_file.writeframes(b''.join(self.frames))
        wav_file.close()
        self.frames = []  # Clear the frames for the next recording session
        
        return torchaudio.load(tmp_output_audio, normalize=True)