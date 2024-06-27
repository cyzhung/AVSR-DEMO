
# 下載預訓練模型

請至 [mpc001/auto_avsr](https://github.com/mpc001/auto_avsr) 下載以下預訓練模型檔案：
- asr_trlrs3vox2_base.pth
- avsr_trlrwlrs2lrs3vox2avsp_base.pth
- vsr_trlrwlrs2lrs3vox2avsp_base.pth
將這些檔案儲存到 AVSR_DEMO/pretrain_model 目錄下。

# 下載面部標誌檢測器
請至[italojs/facial-landmarks-recognition ](https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat)
下載 shape_predictor_68_face_landmarks.dat 文件。將此文件儲存到 AVSR_DEMO/shape_predictor 目錄下。

# 建置環境

```
conda create --name avsr_demo python=3.9
conda activate avsr_demo
pip install -r requirements.txt
```
