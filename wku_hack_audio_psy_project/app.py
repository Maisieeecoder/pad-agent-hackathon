import streamlit as st
import librosa
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

# ========== 解决matplotlib中文乱码（Mac生效） ==========
plt.rcParams["font.family"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号

st.set_page_config(page_title="古风秀女音频匹配系统", layout="wide")
st.title("以声识人，听音观情 | 上传WAV音频，匹配秀女并查看PAD情绪雷达图")

# 读取并清洗数据库
df = pd.read_csv("character_info.csv", encoding="utf-8-sig")
df = df.fillna(0)
df = df.replace([np.inf, -np.inf], 0)

feat_cols = ["f_centroid", "f0_mean", "zcr_mean", "mel_mean", "rms_mean"]
db_vecs = df[feat_cols].to_numpy()

# 上传音频
upload_file = st.file_uploader("上传WAV音频文件", type=["wav"])

if upload_file is not None:
    st.audio(upload_file, format="audio/wav")
    st.info("正在提取音频特征并匹配...")
    try:
        y, sr = librosa.load(upload_file, sr=22050)

        f_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        f0_data = librosa.pyin(y, fmin=50, fmax=500)[0]
        valid_f0 = f0_data[~np.isnan(f0_data)]
        if len(valid_f0) == 0:
            f0 = 150.0
        else:
            f0 = valid_f0.mean()
        zcr = librosa.feature.zero_crossing_rate(y=y).mean()
        mel = librosa.feature.melspectrogram(y=y, sr=sr).mean()
        rms = librosa.feature.rms(y=y).mean()

        input_feat = np.array([f_centroid, f0, zcr, mel, rms])
        input_feat = np.nan_to_num(input_feat, nan=0.0, posinf=0.0, neginf=0.0)

        dists = [euclidean(input_feat, vec) for vec in db_vecs]
        best_idx = np.argmin(dists)
        match_row = df.iloc[best_idx]

        # 展示人物信息
        st.success(f"✅ 匹配成功！匹配人物：{match_row['秀女姓名']}")
        st.write(f"编号：{match_row['编号']}")
        st.write(f"性格类型：{match_row['性格类型']}")
        st.write(f"外貌与穿搭：{match_row['外貌与穿搭']}")
        st.write(f"身份背景：{match_row['身份背景']}")
        st.write(f"PAD：愉悦度={match_row['P(愉悦度)']}｜激活度={match_row['A(激活度)']}｜支配度={match_row['D(支配度)']}")

        # ========== PAD情绪雷达图【古风优化版】 ==========
        labels = ["愉悦度(P)", "激活度(A)", "支配度(D)"]
        values = [match_row["P(愉悦度)"], match_row["A(激活度)"], match_row["D(支配度)"]]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]

        # 画布缩小：figsize=(3,3) 变小，可自行微调，数字越大图越大
        fig, ax = plt.subplots(figsize=(3, 3), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor('#f8f3e6')  # 宣纸米黄色古风背景
        ax.set_facecolor('#f8f3e6')

        # 古风红棕线条+填充
        ax.plot(angles, values, 'o-', linewidth=2, color="#992e2e")
        ax.fill(angles, values, alpha=0.35, color="#c95454")

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylim(-1, 1)
        # 减少刻度数量，解决文字挤在一起
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticklabels(["-1","-0.5","0","0.5","1"], fontsize=9)
        ax.grid(color='#b8a88f', linewidth=0.6) # 网格改为浅棕古风线条

        st.pyplot(fig)

    except Exception as e:
        st.error(f"音频处理失败：{e}")
