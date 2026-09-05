import librosa
import pandas as pd
import os
import numpy as np

audio_dir = "audios"
df = pd.read_csv("character_info.csv", encoding="utf-8-sig")

f_centroid_list = []
f0_mean_list = []
zcr_mean_list = []
mel_mean_list = []
rms_mean_list = []

for idx, row in df.iterrows():
    num = row["编号"]
    name = row["秀女姓名"]
    audio_path = row["音频路径"]
    
    if not os.path.exists(audio_path):
        print(f"⚠️ 文件不存在：{audio_path}，跳过，填充默认值")
        f_centroid_list.append(0.0)
        f0_mean_list.append(150.0)
        zcr_mean_list.append(0.0)
        mel_mean_list.append(0.0)
        rms_mean_list.append(0.0)
        continue
        
    y, sr = librosa.load(audio_path, sr=22050)
    
    f_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    f0_data = librosa.pyin(y, fmin=50, fmax=500)[0]
    valid_f0 = f0_data[~np.isnan(f0_data)]
    
    if len(valid_f0) == 0:
        f0 = 150.0
        print(f"⚠️ {num}_{name}.wav 未检测到基频，使用默认f0=150")
    else:
        f0 = valid_f0.mean()
        
    zcr = librosa.feature.zero_crossing_rate(y=y).mean()
    mel = librosa.feature.melspectrogram(y=y, sr=sr).mean()
    rms = librosa.feature.rms(y=y).mean()
    
    f_centroid_list.append(round(float(f_centroid),4))
    f0_mean_list.append(round(float(f0),4))
    zcr_mean_list.append(round(float(zcr),4))
    mel_mean_list.append(round(float(mel),4))
    rms_mean_list.append(round(float(rms),4))
    
    print(f"✅提取完成 {num}_{name}.wav")

df["f_centroid"] = f_centroid_list
df["f0_mean"] = f0_mean_list
df["zcr_mean"] = zcr_mean_list
df["mel_mean"] = mel_mean_list
df["rms_mean"] = rms_mean_list

df.to_csv("character_info.csv", index=False, encoding="utf-8-sig")
print("\n✅已经把音频特征写回 character_info.csv")
