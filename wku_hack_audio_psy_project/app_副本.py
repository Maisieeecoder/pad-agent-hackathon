import streamlit as st
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="古风秀女音频PAD匹配系统", layout="wide")

# 读取数据表
df = pd.read_csv("character_info.csv", encoding="utf-8-sig")
FEAT_COLS = ["f_centroid","f0_mean","zcr_mean","mel_mean","rms_mean"]

def extract_audio_feature(file_bytes):
    y, sr = librosa.load(BytesIO(file_bytes), sr=22050)
    cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    f0 = np.mean(librosa.yin(y=y,fmin=50,fmax=2000))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    mel = np.mean(librosa.feature.melspectrogram(y=y,sr=sr))
    rms = np.mean(librosa.feature.rms(y=y))
    return np.array([cent,f0,zcr,mel,rms])

def find_most_similar(feat_target):
    df_valid = df.dropna(subset=FEAT_COLS).copy()
    arr_db = df_valid[FEAT_COLS].to_numpy()
    dists = np.linalg.norm(arr_db - feat_target, axis=1)
    best_idx = np.argmin(dists)
    return df_valid.iloc[best_idx]

def draw_pad_radar(p,a,d):
    labels = ["P(愉悦)","A(激活)","D(支配)"]
    vals = [p,a,d]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    vals += vals[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(4,4), subplot_kw={"polar":True})
    ax.plot(angles, vals, "o-", linewidth=2)
    ax.fill(angles, vals, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(-1,1)
    plt.tight_layout()
    return fig

# --------页面UI（交给队友修改这一块做前端美化）--------
st.title("🎙️古风秀女音频PAD情绪匹配系统")
st.markdown("上传一段wav音频，匹配音色最接近的秀女，展示完整人物档案与PAD情绪雷达图")

upload_file = st.file_uploader("上传wav音频文件", type=["wav"])
if upload_file is not None:
    st.audio(upload_file)
    file_bytes = upload_file.read()
    with st.spinner("正在提取音频特征，匹配秀女..."):
        target_feat = extract_audio_feature(file_bytes)
        match_row = find_most_similar(target_feat)

    col1, col2 = st.columns([1.2,1])
    with col1:
        st.subheader(f"✨匹配角色：{match_row['秀女姓名']}")
        img_path = f"character_images/{match_row['编号']}_{match_row['秀女姓名']}.jpg"
        try:
            st.image(img_path, width=220)
        except:
            st.info("未加载到角色头像图片")

        st.markdown(f"""
**性格类型**：{match_row['性格类型']}
**性格关键词**：{match_row['性格关键词']}

**外貌与穿搭**：{match_row['外貌与穿搭']}

**身份背景**：{match_row['身份背景']}

**后续发展与社会地位**：{match_row['后续发展与社会地位']}

**死亡结局(及悲剧主线)**：{match_row['死亡结局(及悲剧主线)']}
""")
    with col2:
        p_val = match_row["P(愉悦度)"]
        a_val = match_row["A(激活度)"]
        d_val = match_row["D(支配度)"]
        st.markdown(f"""
### 📊PAD情绪分数
- P(愉悦度)：`{p_val}`
- A(激活度)：`{a_val}`
- D(支配度)：`{d_val}`
""")
        fig = draw_pad_radar(p_val,a_val,d_val)
        st.pyplot(fig)

st.divider()
st.markdown("项目说明：基于音频声学特征比对，匹配预设10位古风秀女的PAD情绪与人物档案")
