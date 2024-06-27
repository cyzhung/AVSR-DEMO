import os
import time

import hydra
import torch
import torchaudio
import torchvision
import torch.nn.functional as F
import numpy as np

from .datamodule.transforms import AudioTransform, VideoTransform
from .datamodule.av_dataset import cut_or_pad



class InferencePipeline(torch.nn.Module):
    def __init__(self, cfg, detector="retinaface"):
        super(InferencePipeline, self).__init__()
        
        self.modality = cfg.data.modality
        
        self.audio_transform = AudioTransform(subset="test")
        
        if detector == "mediapipe":
            from .preparation.detectors.mediapipe.detector import LandmarksDetector
            from .preparation.detectors.mediapipe.video_process import VideoProcess
            self.landmarks_detector = LandmarksDetector()
            self.video_process = VideoProcess(convert_gray=False)
        elif detector == "retinaface":
            from .preparation.detectors.retinaface.detector import LandmarksDetector
            from .preparation.detectors.retinaface.video_process import VideoProcess
            self.landmarks_detector = LandmarksDetector(device="cuda:0")
            self.video_process = VideoProcess(convert_gray=False)
        self.video_transform = VideoTransform(subset="test")
        if cfg.data.modality in ["audio", "video"]:
            from .lightning import ModelModule
        elif cfg.data.modality == "audiovisual":
            from .lightning_av import ModelModule
        
        self.modelmodule = ModelModule(cfg)
        
        self.modelmodule.model.load_state_dict(torch.load(cfg.pretrained_model_path, map_location=lambda storage, loc: storage))
        self.modelmodule.eval()
    def process_data(self, audio, video, sample_rate):
        current_time = time.localtime()
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
        print(f"{current_time}處理音訊中")
        
        audio = self.audio_process(audio, sample_rate)
        audio = audio.transpose(1, 0)
        audio = self.audio_transform(audio)
        
        current_time = time.localtime()
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
        print(f"{current_time}音訊處理完成")
        
        video = video.permute((0, 3, 1, 2))
        video = self.video_transform(video)
        
        
        current_time = time.localtime()
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
        print(f"{current_time}影像處理完成")
        
        return audio, video
    def load_file(self,audio_fn, video_fn):
        audio = None
        video = None
        audio_fn = os.path.abspath(audio_fn)
        assert os.path.isfile(audio_fn), f"data_filename: {audio_fn} does not exist."
        
        video_fn = os.path.abspath(video_fn)
        assert os.path.isfile(video_fn), f"data_filename: {video_fn} does not exist."


        audio, sample_rate = self.load_audio(audio_fn)
        audio = self.audio_process(audio, sample_rate)
        audio = audio.transpose(1, 0)
        audio = self.audio_transform(audio)

        
        current_time = time.localtime()
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
        print(f"{current_time} 開始載入影像")
        
        video = self.load_video(video_fn)
        
        current_time = time.localtime()
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
        print(f"{current_time}影像載入完成")
        landmarks = self.landmarks_detector(video)

        
        current_time = time.localtime()
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
        print(f"{current_time}偵測嘴唇完成e")
        video = self.video_process(video, landmarks)
        video = torch.tensor(video)
        video = video.permute((0, 3, 1, 2))
        video = self.video_transform(video)
        print(video.shape)
        current_time = time.localtime()
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
        print(f"{current_time}影像處理完成")
        return audio, video
    
    def forward(self, audio,video):
        
        if self.modality == "video":
            with torch.no_grad():
                transcript = self.modelmodule(video)

        elif self.modality == "audio":
            with torch.no_grad():
                transcript = self.modelmodule(audio)

        elif self.modality == "audiovisual":

            rate_ratio = len(audio) // len(video)
            if rate_ratio == 640:
                pass
            else:
                print(f"The ideal video frame rate is set to 25 fps, but the current frame rate ratio, calculated as {len(video)*16000/len(audio):.1f}, which may affect the performance.")
                audio = cut_or_pad(audio, len(video) * 640)
            with torch.no_grad():

                transcript = self.modelmodule(video, audio)

        return transcript

    def load_audio(self, data_filename):
        waveform, sample_rate = torchaudio.load(data_filename, normalize=True)
        return waveform, sample_rate

    def load_video(self, data_filename):
        # 读取视频
        video, _, _ = torchvision.io.read_video(data_filename, pts_unit="sec")
        
        # 计算新的尺寸
        _, H, W, _ = video.shape
        new_H = H // 4
        new_W = W // 4
        
        # 使用interpolate进行下采样
        resized_video = F.interpolate(video.permute(0, 3, 1, 2).float(), size=(new_H, new_W))  # 调整通道位置并转为float
        resized_video = resized_video.permute(0, 2, 3, 1)  # 调整回原来的维度顺序
        
        return resized_video.numpy()

    def audio_process(self, waveform, sample_rate, target_sample_rate=16000):
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, target_sample_rate
            )
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg):
    pipeline = InferencePipeline(cfg)
    transcript = pipeline(cfg.file_path)
    print(f"transcript: {transcript}")


if __name__ == "__main__":
    main()
