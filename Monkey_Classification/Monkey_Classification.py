import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PrimateLens · AI Vision",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Geist+Mono:wght@300;400;500&family=Instrument+Sans:wght@300;400;500&display=swap');

:root {
    --bg:         #08090c;
    --surface:    #0f1218;
    --card:       #131820;
    --border:     #1e2733;
    --border2:    #263040;
    --accent:     #e8f4ff;
    --teal:       #5de6c8;
    --teal-dim:   #2a7a6a;
    --teal-glow:  rgba(93,230,200,0.08);
    --amber:      #f5c842;
    --amber-dim:  #7a6115;
    --red:        #f56060;
    --muted:      #4a5a70;
    --muted2:     #2e3d52;
    --text:       #c8d8ea;
    --text2:      #7a8fa8;
    --white:      #f0f6ff;
}

html, body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Instrument Sans', sans-serif !important;
}

/* scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

/* hide chrome */
#MainMenu, footer
[data-testid="stDecoration"] { display: none !important; }

[data-testid="stFileUploader"] {
    margin-top: 0.3rem;
}

[data-testid="stFileUploaderDropzone"] {
    background: linear-gradient(180deg, #11151c 0%, #0d1117 100%) !important;
    border: 1px solid #222c38 !important;
    border-radius: 16px !important;
    padding: 1.4rem !important;
    transition: all 0.25s ease !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #5de6c8 !important;
    transform: translateY(-1px);
    box-shadow:
        0 0 0 1px rgba(93,230,200,0.08),
        0 10px 24px rgba(0,0,0,0.28);
}

/* upload text */
[data-testid="stFileUploaderDropzoneInstructions"] span {
    color: #d8e4f0 !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: #6f8098 !important;
    font-size: 0.72rem !important;
}

/* button */
[data-testid="stFileUploader"] button {
    background: linear-gradient(180deg, #1a2330 0%, #111827 100%) !important;
    color: #eef6ff !important;
    border: 1px solid #2c3948 !important;
    border-radius: 12px !important;
    padding: 0.48rem 1rem !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.2px !important;
    transition: all 0.2s ease !important;
}

[data-testid="stFileUploader"] button:hover {
    border-color: #5de6c8 !important;
    color: #5de6c8 !important;
    transform: translateY(-1px);
}

[data-testid="stFileUploader"] button:active {
    transform: translateY(0);
}

/* ── Image border ── */
[data-testid="stImage"] img {
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
}

hr { border-color: var(--border) !important; }

/* ── LOGO MARK (CSS) ── */
.logo-wrap {
    display: flex;
    align-items: center;
    gap: 11px;
    padding: 1.4rem 0 1.2rem 0;
}
.logo-mark {
    width: 34px;
    height: 34px;
    position: relative;
    flex-shrink: 0;
}
.logo-mark svg { width: 34px; height: 34px; }
.logo-text {
    font-family: 'Syne', sans-serif;
    font-size: 1.15rem;
    font-weight: 800;
    color: var(--white);
    letter-spacing: -0.3px;
    line-height: 1;
}
.logo-sub {
    font-family: 'Geist Mono', monospace;
    font-size: 0.6rem;
    color: var(--muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 2px;
}

/* ── Sidebar nav links ── */
# .sb-label {
#     font-family: 'Geist Mono', monospace;
#     font-size: 0.6rem;
#     letter-spacing: 2.5px;
#     text-transform: uppercase;
#     color: var(--muted);
#     margin: 1.6rem 0 0.6rem 0;
# }
.sb-info {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.7rem 0.9rem;
    font-size: 0.8rem;
    color: var(--text2);
    line-height: 1.55;
    margin-bottom: 0.8rem;
}
.sb-chip {
    display: inline-block;
    background: var(--card);
    border: 1px solid var(--border2);
    color: var(--text2);
    font-size: 0.65rem;
    font-family: 'Geist Mono', monospace;
    padding: 0.22rem 0.6rem;
    border-radius: 4px;
    line-height: 1.4;
}
.sb-footer {
    font-family: 'Geist Mono', monospace;
    font-size: 0.6rem;
    color: var(--muted);
    letter-spacing: 1px;
    padding: 0.8rem 0 0.3rem 0;
    border-top: 1px solid var(--border);
    margin-top: 1rem;
}

/* ── Page header ── */
.page-header {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    margin-bottom: 2rem;
    padding-bottom: 1.2rem;
    border-bottom: 1px solid var(--border);
}
.page-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.7rem;
    font-weight: 800;
    color: var(--white);
    letter-spacing: -0.5px;
    line-height: 1;
    margin: 0;
}
.page-sub {
    font-size: 0.8rem;
    color: var(--text2);
    margin-top: 0.35rem;
    letter-spacing: 0.3px;
}
.status-pill {
    display: flex;
    align-items: center;
    gap: 6px;
    background: var(--teal-glow);
    border: 1px solid var(--teal-dim);
    border-radius: 20px;
    padding: 0.3rem 0.85rem;
    font-family: 'Geist Mono', monospace;
    font-size: 0.65rem;
    color: var(--teal);
    letter-spacing: 1px;
    text-transform: uppercase;
}
.status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--teal);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
}

/* ── Section label ── */
.sec-label {
    font-family: 'Geist Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.75rem;
}

/* ── Top result card ── */
.result-top {
    background: var(--card);
    border: 1px solid var(--border2);
    border-radius: 14px;
    padding: 1.5rem 1.6rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.result-top::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--teal), transparent);
}
.result-name {
    font-family: 'Syne', sans-serif;
    font-size: 1.45rem;
    font-weight: 800;
    color: var(--white);
    letter-spacing: -0.3px;
    margin-bottom: 0.2rem;
}
.result-conf-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 1rem;
}
.result-conf-pct {
    font-family: 'Geist Mono', monospace;
    font-size: 1.1rem;
    color: var(--teal);
    font-weight: 500;
}
.conf-track {
    flex: 1;
    margin: 0 1rem;
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    background: var(--teal);
    border-radius: 2px;
}

/* ── Runner-up card ── */
.runner-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1rem;
}
.runner-name { font-size: 0.88rem; color: var(--text); }
.runner-pct {
    font-family: 'Geist Mono', monospace;
    font-size: 0.8rem;
    color: var(--amber);
}

/* ── Stat boxes ── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.7rem;
    margin-top: 0.9rem;
}
.stat-box {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.8rem 0.9rem;
    text-align: center;
}
.stat-val {
    font-family: 'Geist Mono', monospace;
    font-size: 1.05rem;
    color: var(--accent);
    font-weight: 500;
}
.stat-lbl {
    font-family: 'Geist Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 0.2rem;
}

/* ── Breakdown ── */
.breakdown-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
}
.bk-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 0.55rem;
    font-size: 0.78rem;
}
.bk-rank {
    font-family: 'Geist Mono', monospace;
    font-size: 0.62rem;
    color: var(--muted);
    width: 14px;
}
.bk-name {
    width: 170px;
    color: var(--text);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.bk-track {
    flex: 1;
    height: 5px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
}
.bk-fill { height: 100%; border-radius: 3px; }
.bk-pct {
    font-family: 'Geist Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    width: 42px;
    text-align: right;
}

/* ── Empty state ── */
.empty-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 6rem 2rem;
    border: 1px dashed var(--border2);
    border-radius: 16px;
    background: var(--surface);
    text-align: center;
}
.empty-icon {
    width: 56px;
    height: 56px;
    background: var(--card);
    border: 1px solid var(--border2);
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 1.2rem;
}
.empty-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: var(--text2);
    margin-bottom: 0.4rem;
}
.empty-sub { font-size: 0.8rem; color: var(--muted); }
.empty-chips {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 0.35rem;
    max-width: 520px;
    margin-top: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Mantled Howler", "Patas Monkey", "Bald Uakari", "Japanese Macaque",
    "Pygmy Marmoset", "White Headed Capuchin", "Silvery Marmoset",
    "Common Squirrel Monkey", "Black Headed Night Monkey", "Nilgiri Langur"
]

BAR_COLORS = [
    "#5de6c8", "#f5c842", "#f56060", "#7ec8f5",
    "#c8a0f5", "#5de6c8", "#f5c842", "#f56060", "#7ec8f5", "#c8a0f5"
]

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOAD
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_classifier():
    return tf.keras.models.load_model('Monkey_Species2.keras')

model = load_classifier()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo
    st.markdown("""
    <div class="logo-wrap">
        <div class="logo-mark">
            <svg viewBox="0 0 34 34" fill="none" xmlns="http://www.w3.org/2000/svg">
                <rect width="34" height="34" rx="9" fill="#0f1e30"/>
                <circle cx="17" cy="14" r="7" stroke="#5de6c8" stroke-width="1.5" fill="none"/>
                <circle cx="17" cy="14" r="3" fill="#5de6c8" opacity="0.9"/>
                <line x1="17" y1="21" x2="17" y2="27" stroke="#5de6c8" stroke-width="1.5" stroke-linecap="round"/>
                <line x1="13" y1="24" x2="21" y2="24" stroke="#5de6c8" stroke-width="1.5" stroke-linecap="round"/>
                <circle cx="10" cy="9" r="2" stroke="#5de6c8" stroke-width="1" fill="none" opacity="0.5"/>
                <circle cx="24" cy="9" r="2" stroke="#5de6c8" stroke-width="1" fill="none" opacity="0.5"/>
            </svg>
        </div>
        <div>
            <div class="logo-text">PrimateLens</div>
            <div class="logo-sub">AI Vision · v2.0</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        " Upload Image",
        type=['jpg', 'jpeg', 'png'],
        label_visibility="collapsed"
    )

    st.markdown(
        '<div class="sb-info">Upload a clear photo of a primate. '
        'Front-facing images with natural lighting yield the highest accuracy.</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="sb-label">Species Library</div>', unsafe_allow_html=True)
    chips_html = '<div class="chips-wrap">' + "".join(f'<span class="sb-chip">{n}</span>' for n in CLASS_NAMES) + '</div>'
    st.markdown(chips_html, unsafe_allow_html=True)

    st.markdown(
        '<div class="sb-footer">PrimateLens · CNN v2 · Code_Forge</div>',
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <div>
        <div class="page-title">Species Analysis</div>
        <div class="page-sub">Neural classification · 10 primate species · 224 × 224 input</div>
    </div>
    <div class="status-pill">
        <span class="status-dot"></span>
        Model Ready
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    # Inference
    img_resize = image.resize((224, 224))
    img_array  = np.array(img_resize) / 255.0
    img_array  = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing…"):
        predictions = model.predict(img_array)
        score = predictions[0]

    top_idx   = int(np.argmax(score))
    result    = CLASS_NAMES[top_idx]
    conf      = float(score[top_idx]) * 100
    top2_idx  = int(np.argsort(score)[-2])
    top2_name = CLASS_NAMES[top2_idx]
    top2_conf = float(score[top2_idx]) * 100
    entropy   = float(-np.sum(score * np.log(score + 1e-9)))
    w, h      = image.size

    col_l, col_r = st.columns([1, 1], gap="large")

    # ── Left: image + stats
    with col_l:
        st.markdown('<div class="sec-label">Input Image</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True, caption=uploaded_file.name)

        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-box">
                <div class="stat-val">{w}×{h}</div>
                <div class="stat-lbl">Resolution</div>
            </div>
            <div class="stat-box">
                <div class="stat-val">10</div>
                <div class="stat-lbl">Classes</div>
            </div>
            <div class="stat-box">
                <div class="stat-val">{entropy:.2f}</div>
                <div class="stat-lbl">Entropy</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Right: results
    with col_r:
        st.markdown('<div class="sec-label">Classification Result</div>', unsafe_allow_html=True)

        # Primary result
        fill_w = int(conf)
        st.markdown(f"""
        <div class="result-top">
            <div style="font-family:'Geist Mono',monospace;font-size:0.6rem;letter-spacing:2px;
                        text-transform:uppercase;color:var(--muted);margin-bottom:0.5rem;">
                Top Prediction
            </div>
            <div class="result-name">{result}</div>
            <div class="result-conf-row">
                <div class="result-conf-pct">{conf:.1f}%</div>
                <div class="conf-track">
                    <div class="conf-fill" style="width:{fill_w}%;"></div>
                </div>
                <div style="font-family:'Geist Mono',monospace;font-size:0.65rem;color:var(--muted);">confidence</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Runner-up
        st.markdown(f"""
        <div class="runner-card">
            <div>
                <div style="font-family:'Geist Mono',monospace;font-size:0.58rem;
                            letter-spacing:2px;text-transform:uppercase;color:var(--muted);margin-bottom:3px;">
                    Runner-up
                </div>
                <div class="runner-name">{top2_name}</div>
            </div>
            <div class="runner-pct">{top2_conf:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence breakdown
        sorted_idx = np.argsort(score)[::-1]
        rows = []
        for rank, idx in enumerate(sorted_idx):
            pct    = float(score[idx]) * 100
            bar_w  = int(pct)
            color  = BAR_COLORS[rank % len(BAR_COLORS)]
            bold   = "color:var(--white);font-weight:500;" if idx == top_idx else ""
            rows.append(
                f'<div class="bk-row">'
                f'<div class="bk-rank">{rank+1}</div>'
                f'<div class="bk-name" style="{bold}">{CLASS_NAMES[idx]}</div>'
                f'<div class="bk-track"><div class="bk-fill" style="width:{bar_w}%;background:{color};"></div></div>'
                f'<div class="bk-pct">{pct:.1f}%</div>'
                f'</div>'
            )

        st.markdown(
            '<div class="sec-label" style="margin-top:1rem;">Confidence Breakdown</div>'
            '<div class="breakdown-card">'
            + "".join(rows) +
            '</div>',
            unsafe_allow_html=True
        )

else:
    # ── Empty state
    chips_html = "".join(f'<span class="sb-chip">{n}</span>' for n in CLASS_NAMES)
    st.markdown(f"""
    <div class="empty-wrap">
        <div class="empty-icon">
            <svg width="26" height="26" viewBox="0 0 26 26" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="13" cy="11" r="6" stroke="#5de6c8" stroke-width="1.4" fill="none"/>
                <circle cx="13" cy="11" r="2.5" fill="#5de6c8" opacity="0.8"/>
                <line x1="13" y1="17" x2="13" y2="22" stroke="#5de6c8" stroke-width="1.4" stroke-linecap="round"/>
                <line x1="10" y1="20" x2="16" y2="20" stroke="#5de6c8" stroke-width="1.4" stroke-linecap="round"/>
            </svg>
        </div>
        <div class="empty-title">No image uploaded</div>
        <div class="empty-sub">Upload a primate photo in the sidebar to begin classification.</div>
        <div class="empty-chips">{chips_html}</div>
    </div>
    """, unsafe_allow_html=True)