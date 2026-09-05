import streamlit as st
from streamlit_extras.stylable_container import stylable_container
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
from io import BytesIO
import os
import base64
import streamlit.components.v1 as components

# 页面状态标记，控制是否展示匹配结果
if "show_result" not in st.session_state:
    st.session_state.show_result = False

st.set_page_config(page_title="古韵声踪 · 秀女PAD情绪档案", layout="wide")

# 初始化session_state变量
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

# 你之前的视频切换状态也要初始化，一并加上
if "show_destiny_video" not in st.session_state:
    st.session_state["show_destiny_video"] = False

# ==========在这里粘贴CSS！！==========
st.markdown("""
<style>
/* 查看人物命运 btn_show_end */
.st-key-btn_show_end .stButton > button {
    background-color:#911818 !important;
    border:none !important;
    padding: 8px 20px !important;
    font-size:16px !important;
}
.st-key-btn_show_end .stButton > button *{
    color:#ffffff !important;
    font-weight:bold !important;
}
.st-key-btn_show_end .stButton > button:hover {
    background-color:#6e1010 !important;
}

/* 返回角色立绘短片 btn_back_ending */
.st-key-btn_back_ending .stButton > button {
    background-color:#911818 !important;
    border:none !important;
    padding: 8px 20px !important;
    font-size:16px !important;
}
.st-key-btn_back_ending .stButton > button *{
    color:#ffffff !important;
    font-weight:bold !important;
}
.st-key-btn_back_ending .stButton > button:hover {
    background-color:#6e1010 !important;
}
</style>
""", unsafe_allow_html=True)


def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_base64 = get_base64_of_bin_file("assets/bg_girl_xuanzhi.png")
print("图片路径读取成功")

# 古风宣纸背景 + 轻微浮动动画
st.markdown(f"""
<style>
/* 外层页面容器 */
[data-testid="stAppViewContainer"] {{
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  overflow: hidden;
}}

/* 伪元素承载背景图，动画绑定在这里 */
[data-testid="stAppViewContainer"]::before {{
  content: "";
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-image: url("data:image/png;base64,{img_base64}");
  background-size: cover;
  background-position: center;
  z-index: -1;
  animation: bgMove 6s ease-in-out infinite alternate;
}}

.block-container {{
  background-color: transparent !important;
}}

@keyframes bgMove {{
  0% {{
    background-position: 48% 48%;
  }}
  100% {{
    background-position: 52% 52%;
  }}
}}

/* ========== 修复文字颜色 ========== */

/* 👉 1. 匹配结果卡片内 情绪类型、性格关键词（重点修复！） */
div[data-testid="stVerticalBlock"] div[data-testid="stMarkdownContainer"] p {{
    color: #681c1c !important;
    opacity: 1 !important;
}}

/* 👉 2. PAD指标卡片 stMetric 数值+标签（已经生效，保留） */
div[data-testid="stMetric"] * {{
    color: #681c1c !important;
}}

/* 👉3. 故事档案板块，文本区域里面的文字 */
.stTextArea textarea {{
    color: #681c1c !important;
}}
.stTextArea > label, .stTextInput > label {{
    color: #681c1c !important;
}}

/* 👉4. 录音、上传文件文字 */
.stAudioRecorder p, .stAudioRecorder span, .stFileUploader span {{
    color: #681c1c !important;
}}

/* 👉5. Plotly雷达图内所有文字、坐标轴标签A P D */
.stPlotlyChart svg text {{
    fill: #681c1c !important;
}}

/* 👉6. 卡片底色加深，提高对比度 */
section[data-testid="stVerticalBlockBorderWrapper"], div[data-testid="stHorizontalBlock"] > div {{
    background-color: rgba(255,252,245,0.88) !important;
    border-radius: 10px !important;
}}

/* 👉7. “未加载到角色立绘图片”提示文字 */
div.stAlert p {{
    color: #681c1c !important;
}}
/* 专门修复档案卡片内白色文字：情绪类型、性格关键词、故事正文 */
div[data-testid="stVerticalBlockBorderWrapper"] .stMarkdown p {{
    color: #681c1c !important;
    opacity: 1 !important;
}}
div[data-testid="stVerticalBlockBorderWrapper"] .stMarkdown span {{
    color: #681c1c !important;
    opacity: 1 !important;
}}



</style>
""", unsafe_allow_html=True)


# 读取数据表 —— 以 app.py 所在目录为基准定位 csv
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_CSV_PATH = os.path.join(_BASE_DIR, "character_info.csv")
df = pd.read_csv(_CSV_PATH, encoding="utf-8-sig")
VIDEO_FOLDER = os.path.join(_BASE_DIR, "video")
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
    labels = ["P","A","D"]
    vals = [p,a,d]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    vals += vals[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(3,3), subplot_kw={"polar":True})
    ax.plot(angles, vals, "o-", linewidth=2)
    ax.fill(angles, vals, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticks([-1,-0.5,0,0.5,1])
    ax.set_yticklabels(["-1","-0.5","0","0.5","1"])
    ax.set_ylim(-1,1)
    plt.tight_layout()
    return fig

# -------- 页面UI（古风宫廷视觉升级）--------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Ma+Shan+Zheng&family=Noto+Serif+SC:wght@400;700;900&display=swap');

/* ===== 隐藏侧边栏 ===== */
[data-testid="stSidebar"] { display: none !important; }
[data-testid="stSidebarCollapseButton"] { display: none !important; }
button[kind="header"] { display: none !important; }

/* ===== 主背景：宣纸纹理 + 多层宫廷氛围 ===== */
.stApp {
    background:
        radial-gradient(ellipse at 20% 10%, rgba(201,163,91,0.10) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 90%, rgba(110,35,30,0.08) 0%, transparent 50%),
        linear-gradient(135deg, #FBF6EC 0%, #F5EBD7 40%, #EDE0C8 100%);
    background-attachment: fixed;
}

/* 金色粒子漂浮 */
@keyframes floatParticle {
    0%   { transform: translateY(0) translateX(0); opacity: 0; }
    20%  { opacity: 0.6; }
    80%  { opacity: 0.4; }
    100% { transform: translateY(-120px) translateX(15px); opacity: 0; }
}
.gold-particle {
    position: fixed;
    width: 4px; height: 4px;
    background: radial-gradient(circle, #D8B56A 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
    animation: floatParticle 12s ease-in-out infinite;
}

/* ===== 宫廷牌匾主标题 ===== */
@keyframes titleFadeIn {
    0%   { opacity: 0; transform: scale(0.92) translateY(-10px); }
    100% { opacity: 1; transform: scale(1) translateY(0); }
}
@keyframes titleGlow {
    0%, 100% { text-shadow: 1px 1px 3px rgba(201,163,91,0.35), 0 0 8px rgba(216,181,106,0.20); }
    50%      { text-shadow: 1px 1px 3px rgba(201,163,91,0.50), 0 0 18px rgba(216,181,106,0.40); }
}

.title-plaque {
    position: relative;
    text-align: center;
    padding: 28px 60px 22px;
    margin: 10px auto 6px;
    max-width: 900px;
    background: linear-gradient(180deg, rgba(123,45,38,0.06) 0%, rgba(201,163,91,0.08) 100%);
    border-top: 2px solid #C9A35B;
    border-bottom: 2px solid #C9A35B;
    border-left: 1px solid rgba(201,163,91,0.4);
    border-right: 1px solid rgba(201,163,91,0.4);
    animation: titleFadeIn 1.5s ease-out;
}
.title-plaque::before, .title-plaque::after {
    content: "☁";
    position: absolute;
    top: 50%;
    transform: translateY(-50%);
    font-size: 2rem;
    color: rgba(201,163,91,0.35);
}
.title-plaque::before { left: 12px; }
.title-plaque::after  { right: 12px; }

.main-title {
    font-family: 'Ma Shan Zheng', 'Noto Serif SC', serif;
    font-size: 3.8rem;
    font-weight: 900;
    color: #6E231E;
    letter-spacing: 0.08em;
    line-height: 1.2;
    animation: titleGlow 6s ease-in-out infinite;
    text-shadow:
        -1px -1px 0 #C9A35B,
         1px  1px 0 #C9A35B,
         0 0 12px rgba(216,181,106,0.3);
}
.sub-title {
    font-family: 'Noto Serif SC', serif;
    font-size: 1.0rem;
    color: #5C4033;
    letter-spacing: 0.15em;
    margin-top: 10px;
    line-height: 2;
}
.sub-title-hint {
    font-size: 0.82rem;
    color: #8C7B6B;
    margin-top: 4px;
}

/* ===== 祥云纹分割线 ===== */
.cloud-divider {
    text-align: center;
    margin: 18px 0;
    color: #C9A35B;
    font-size: 1.1rem;
    letter-spacing: 1em;
    opacity: 0.5;
}
.cloud-divider::before, .cloud-divider::after {
    content: "─────────";
    color: rgba(201,163,91,0.3);
    letter-spacing: normal;
    margin: 0 8px;
}

/* ===== 流程导航 ===== */
@keyframes stepPulse {
    0%, 100% { box-shadow: 0 0 6px rgba(201,163,91,0.3); }
    50%      { box-shadow: 0 0 14px rgba(216,181,106,0.6); }
}
.flow-nav {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
    margin: 10px 0 6px;
    font-size: 0.85rem;
}
.flow-step {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 5px 14px;
    border: 1px solid rgba(201,163,91,0.4);
    border-radius: 20px;
    background: rgba(245,235,215,0.6);
    color: #6E231E;
    font-weight: 600;
}
.flow-step.active {
    background: linear-gradient(135deg, #6E231E, #7B2D26);
    color: #F5EBD7;
    border-color: #C9A35B;
    animation: stepPulse 2.5s ease-in-out infinite;
}
.flow-arrow {
    color: #C9A35B;
    font-size: 0.9rem;
}

/* ===== 区块标题 ===== */
.section-header {
    font-family: 'Noto Serif SC', serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: #6E231E;
    border-left: 4px solid #C9A35B;
    padding: 4px 0 4px 12px;
    margin: 0.8rem 0 1rem;
    letter-spacing: 0.06em;
}

/* ===== 卷轴上传区域 ===== */
.scroll-zone {
    background: linear-gradient(180deg, rgba(251,246,236,0.85) 0%, rgba(243,232,210,0.75) 100%);
    border: 2px solid #C9A35B;
    border-radius: 4px;
    padding: 20px;
    margin-bottom: 14px;
    position: relative;
    box-shadow: 0 3px 12px rgba(110,35,30,0.08), inset 0 0 30px rgba(201,163,91,0.05);
    transition: all 0.4s ease;
}
.scroll-zone::before {
    content: "☁ ☁ ☁";
    position: absolute;
    top: -2px; left: 50%;
    transform: translateX(-50%);
    background: #FBF6EC;
    padding: 0 12px;
    color: rgba(201,163,91,0.4);
    font-size: 0.75rem;
    letter-spacing: 0.3em;
}

/* ===== 结果卷宗卡片 ===== */
@keyframes cardFadeUp {
    0%   { opacity: 0; transform: translateY(15px); }
    100% { opacity: 1; transform: translateY(0); }
}
.result-card {
    background: linear-gradient(180deg, rgba(255,253,247,0.88) 0%, rgba(245,235,215,0.72) 100%);
    border: 1px solid rgba(201,163,91,0.35);
    border-left: 3px solid #7B2D26;
    border-radius: 6px;
    padding: 16px 18px;
    margin-bottom: 14px;
    box-shadow: 0 2px 10px rgba(110,35,30,0.07);
    animation: cardFadeUp 0.6s ease-out;
    position: relative;
}
.result-card::after {
    content: "档案";
    position: absolute;
    top: 8px; right: 10px;
    font-size: 0.6rem;
    color: rgba(123,45,38,0.25);
    border: 1px solid rgba(123,45,38,0.2);
    border-radius: 3px;
    padding: 1px 5px;
}

/* ===== 朱砂印 ===== */
.seal {
    display: inline-block;
    background: #6E231E;
    color: #F5EBD7;
    font-family: 'Ma Shan Zheng', serif;
    font-size: 0.7rem;
    padding: 4px 10px;
    border-radius: 3px;
    letter-spacing: 0.1em;
    transform: rotate(-3deg);
    box-shadow: 0 1px 4px rgba(110,35,30,0.3);
}
/* ===== PAD 数值卡片 ===== */
.pad-row {
    display: flex;
    gap: 10px;
    margin: 8px 0;
}
.pad-box {
    flex: 1;
    text-align: center;
    background: rgba(245,235,215,0.5);
    border: 1px solid rgba(201,163,91,0.3);
    border-radius: 5px;
    padding: 8px 4px;
}
.pad-letter {
    font-size: 1.3rem;
    font-weight: 900;
    color: #6E231E;
}
.pad-label {
    font-size: 0.7rem;
    color: #5C4033;
}
.pad-val {
    font-size: 1.0rem;
    font-weight: 700;
    color: #56756B;
    margin-top: 2px;
}

/* ===== 令牌按钮 ===== */
.stButton > button {
    background: linear-gradient(135deg, #6E231E 0%, #7B2D26 100%) !important;
    color: #F5EBD7 !important;
    border: 2px solid #C9A35B !important;
    border-radius: 4px !important;
    font-family: 'Noto Serif SC', serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    box-shadow: 0 2px 6px rgba(110,35,30,0.25) !important;
    transition: all 0.35s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #7B2D26, #8C3830) !important;
    border: 2px solid #D8B56A !important;
    box-shadow: 0 0 16px rgba(216,181,106,0.5), 0 4px 12px rgba(110,35,30,0.3) !important;
    transform: translateY(-3px) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ===== 分割线 ===== */
hr {
    border: none !important;
    border-top: 1px solid rgba(201,163,91,0.35) !important;
    margin: 0.9rem 0 !important;
}

/* ===== 提示框 ===== */
.mic-hint {
    text-align: center;
    font-size: 0.8rem;
    color: #8C7B6B;
    background: rgba(201,163,91,0.08);
    border: 1px dashed rgba(201,163,91,0.4);
    border-radius: 5px;
    padding: 8px 12px;
    margin: 8px 0;
}
.empty-archives {
    text-align: center;
    color: #8C7B6B;
    padding: 30px 20px;
}
.empty-archives .mirror {
    font-size: 2.5rem;
    color: rgba(201,163,91,0.3);
    margin-bottom: 10px;
}

/* ===== 底部说明 ===== */
.footer-scroll {
    text-align: center;
    color: #5C4033;
    font-size: 0.82rem;
    background: rgba(201,163,91,0.06);
    border-top: 1px solid rgba(201,163,91,0.3);
    border-bottom: 1px solid rgba(201,163,91,0.3);
    padding: 14px 20px;
    margin-top: 10px;
}

/* ===== 响应式 ===== */
@media (max-width: 768px) {
    .main-title { font-size: 2.2rem !important; }
    .title-plaque { padding: 18px 30px 14px; }
    .pad-row { flex-direction: column; }
}

/* ===== 减少动画偏好 ===== */
@media (prefers-reduced-motion: reduce) {
    * { animation: none !important; transition: none !important; }
}
</style>
""", unsafe_allow_html=True)

# 金色粒子（5个，不同延迟避免同步）
particle_html = "".join([
    f'<div class="gold-particle" style="left:{x}%;bottom:10%;animation-delay:{d}s;animation-duration:{dur}s;"></div>'
    for x, d, dur in [(15,0,14),(35,3,16),(55,1.5,13),(72,4.5,15),(88,2,17)]
])
st.markdown(f'<div style="position:fixed;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:0;overflow:hidden;">{particle_html}</div>', unsafe_allow_html=True)

# ===== 顶部宫廷牌匾标题 =====
st.markdown("""
<div class="title-plaque">
    <div class="main-title" style="white-space: nowrap;">古韵声踪 · 秀女PAD情绪档案</div>
    <div class="sub-title">声传千年古韵 · 心测一缕情丝</div>
    <div class="sub-title-hint">轻启朱唇，以声音情绪寻觅与你命格最契合的秀女</div>
</div>
""", unsafe_allow_html=True)

# ===== 流程导航 =====
has_result = 'match_row' in st.session_state
st.markdown(f"""
<div class="flow-nav">
    <span class="flow-step active">① 上传声音</span>
    <span class="flow-arrow">↓</span>
    <span class="flow-step {'active' if has_result else ''}">② 分析PAD</span>
    <span class="flow-arrow">↓</span>
    <span class="flow-step {'active' if has_result else ''}">③ 匹配秀女</span>
    <span class="flow-arrow">↓</span>
    <span class="flow-step {'active' if has_result else ''}">④ 情绪档案</span>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="cloud-divider">祥 ☁ 云 ☁ 纹</div>', unsafe_allow_html=True)

# ===== 左右分栏 [2 : 1] =====
col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown('<div class="section-header">🎙️ 音频上传 · 操作区</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="mic-hint">
        📜 上传一段 <b>.wav</b> 音频，或直接现场录音，系统将以声学特征为尺，从十位古风秀女中寻觅与你音色最契合的那一位<br>
        支持 .WAV 格式 · 最大 200MB
    </div>
    """, unsafe_allow_html=True)

    # 音频来源切换：现场录音 / 上传WAV
audio_source = st.radio("选择音频来源", ["🎙️ 现场录音", "📁 上传WAV文件"], horizontal=True)
audio_data = None
# 用"包含文字"判断，彻底避开emoji编码不一样的坑！不管emoji是什么都能匹配
if "现场录音" in audio_source:
    # 去掉动态key，保证录音按钮一定能显示出来
    audio_data = st.audio_input("点击麦克风录制声音")
else:
    upload_file = st.file_uploader(
        "选择 wav 音频文件",
        type=["wav"],
        key=f"uploader_{st.session_state['uploader_key']}",
        label_visibility="collapsed"
    )
    audio_data = upload_file


if audio_data is not None:
    # 创建左右双栏，左55% 右45%，比例可以调
    col_left, col_right = st.columns([0.62, 0.38])

    with col_left:
        # ========== 左栏：音频播放器 ==========
        st.audio(audio_data)
        file_bytes = audio_data.read()

        # 特征提取 + 匹配（计算逻辑放左栏）
        with st.spinner("正在聆听你的声音，解析情绪..."):
            target_feat = extract_audio_feature(file_bytes)
            match_row = find_most_similar(target_feat)
            st.session_state['match_row'] = match_row

            char_pinyin = match_row["pinyin"]
            st.session_state.show_result = True
            intro_video_name = f"role_{char_pinyin}.mp4"
            intro_video_path = os.path.join(VIDEO_FOLDER, intro_video_name)
            if "show_destiny_video" not in st.session_state:
             st.session_state["show_destiny_video"] = False
            #========正确的视频切换逻辑（依靠show_destiny_video判断）========
            if st.session_state["show_destiny_video"] == False:
                if os.path.exists(intro_video_path):
                    st.subheader("角色立绘短片")
                    st.video(intro_video_path, format="video/mp4")
                else:
                    st.info(f"缺失角色主视频: {intro_video_name}")
                # 查看人物命运按钮
                col_btn1, col_btn2, col_btn3 = st.columns([1,2,1])
                with col_btn2:
                    if st.button("🎬 查看人物命运", key="btn_show_end"):
                        st.session_state["show_destiny_video"] = True
                        st.rerun()
            else:
                # 人物结局短片
                destiny_video_name = f"role_{char_pinyin}_ending.mp4"
                destiny_video_path = os.path.join(VIDEO_FOLDER, destiny_video_name)
                st.markdown("### 人物命运终章")
                if os.path.exists(destiny_video_path):
                    st.video(destiny_video_path, format="video/mp4")
                else:
                    st.warning(f"暂无该人物命运短片：{destiny_video_name}")
                # 返回立绘短片按钮
                col_btn1, col_btn2, col_btn3 = st.columns([1,2,1])
                with col_btn2:
                    if st.button("🎥 返回角色立绘短片", key="btn_back_ending"):
                        st.session_state["show_destiny_video"] = False
                        st.rerun()


                
        # ========== 左栏：角色视频（spinner外面，匹配完才显示）==========
    

    with col_right:
        # ========== 右栏：所有匹配结果展示 ==========
            st.success(f"✅ 命运已开启！最契合的秀女：**{match_row['秀女姓名']}**")
            st.markdown('<div class="cloud-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-header">📋 匹配结果</div>', unsafe_allow_html=True)
            if 'match_row' in st.session_state:    
                match_row = st.session_state['match_row']
                p_val = match_row["P(愉悦度)"]
                a_val = match_row["A(激活度)"]
                d_val = match_row["D(支配度)"]

                # 秀女档案卷宗
                st.markdown(f"""
        <div class="result-card"
        style="color:#681c1c;">
        <span class="seal">秀女档案</span>
        <h3 style="color:#6E231E; margin:8px 0 4px 0; font-family:'Ma Shan Zheng',serif; font-size:1.8rem;">{match_row['秀女姓名']}</h3>
        <b>情绪类型</b>　{match_row['性格类型']}<br>
        <b>性格关键词</b>　{match_row['性格类型']}
        </div>
        """, unsafe_allow_html=True)

                img_path = f"character_images/{match_row['编号']}_{match_row['秀女姓名']}.jpg"
                try:
                    st.image(img_path, width=200)
                except:
                    pass
                # PAD 三才盘数值
                st.markdown(f"""
        <div class="result-card">
        <b>📊 PAD 情绪三才盘</b>
        <div class="pad-row">
            <div class="pad-box">
                <div class="pad-letter">P</div>
                <div class="pad-label">愉悦度</div>
                <div class="pad-val">{p_val}</div>
            </div>
            <div class="pad-box">
                <div class="pad-letter">A</div>
                <div class="pad-label">唤醒度</div>
                <div class="pad-val">{a_val}</div>
            </div>
            <div class="pad-box">
                <div class="pad-letter">D</div>
                <div class="pad-label">支配度</div>
                <div class="pad-val">{d_val}</div>
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

                # 复用原有雷达图绘图
                fig = draw_pad_radar(p_val, a_val, d_val)
                st.pyplot(fig)

                # 人物命运档案
                st.markdown(f"""
        <div class="result-card"
        style="color:#681c1c;">
        <b>🌸 外貌与穿搭</b><br>
        {match_row['外貌与穿搭']}
        <hr>
        <b>🏛️ 身份背景</b><br>
        {match_row['身份背景']}
        <hr>
        <b>📈 后续发展与社会地位</b><br>
        {match_row['身份背景']}
        <hr>
        <b>🍂 死亡结局（及悲剧主线）</b><br>
        {match_row['死亡结局(及悲剧主线)']}
        </div>
        """, unsafe_allow_html=True)
            else:
                st.markdown("""
        <div class="result-card empty-archives">
        <div class="mirror">🪞</div>
        <b style="color:#6E231E;">尚未开启情绪命运</b><br><br>
        请在左侧上传 <code>.wav</code> 音频文件，或直接现场录音<br>
        系统将通过声音情绪特征<br>
        匹配属于你的秀女 PAD 情绪档案
        </div>
        """, unsafe_allow_html=True)
            
            
        # 操作按钮（令牌风格）


st.markdown('<div class="cloud-divider">❖ ☁ ❖ ☁ ❖</div>', unsafe_allow_html=True)
