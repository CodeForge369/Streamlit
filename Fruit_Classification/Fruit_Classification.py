import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Classy-AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS: Claude/ChatGPT-inspired clean dark UI ─────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg-primary:    #0E1117;
    --bg-secondary:  #2f2f2f;
    --bg-tertiary:   #3a3a3a;
    --border:        rgba(255,255,255,0.1);
    --border-hover:  rgba(255,255,255,0.2);
    --text-primary:  #ececec;
    --text-secondary:#8e8ea0;
    --text-muted:    #5a5a72;
    --accent:        #10a37f;
    --accent-hover:  #1a7f64;
    --radius-sm:     8px;
    --radius-md:     12px;
    --transition:    0.18s ease;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"] {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', system-ui, sans-serif !important;
    font-size: 15px;
    line-height: 1.6;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #171717 !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding: 1rem 0.75rem !important;
}

.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.5rem 0.75rem 1.2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1rem;
}

.sidebar-logo-icon {
    width: 32px; height: 32px;
    background: var(--accent);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
}

.sidebar-logo-name {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: -0.01em;
}

.sidebar-section {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-muted);
    padding: 0.4rem 0.75rem;
    margin-top: 1rem;
    margin-bottom: 0.25rem;
}

.sidebar-item {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.55rem 0.75rem;
    border-radius: var(--radius-sm);
    font-size: 0.88rem;
    color: var(--text-secondary);
    cursor: pointer;
    transition: background var(--transition), color var(--transition);
    margin-bottom: 2px;
}

.sidebar-item:hover { background: var(--bg-secondary); color: var(--text-primary); }
.sidebar-item.active { background: var(--bg-secondary); color: var(--text-primary); }
.sidebar-item-icon { font-size: 1rem; opacity: 0.7; }

.model-badge {
    margin: 1.5rem 0.75rem 0;
    padding: 0.65rem 0.8rem;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    font-size: 0.78rem;
    color: var(--text-secondary);
}

.model-badge-title {
    font-weight: 600;
    color: var(--text-primary);
    font-size: 0.82rem;
    margin-bottom: 0.2rem;
}

.model-badge-sub { font-size: 0.72rem; color: var(--text-muted); }

.online-dot {
    display: inline-block;
    width: 6px; height: 6px;
    background: var(--accent);
    border-radius: 50%;
    margin-right: 5px;
    box-shadow: 0 0 6px var(--accent);
}

/* ── Main ── */
[data-testid="block-container"] {
    max-width: 920px !important;
    padding: 2rem 2.5rem !important;
    margin: 0 auto !important;
}

.page-heading { margin-bottom: 1.75rem; }
.page-heading h1 {
    font-size: 1.6rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: var(--text-primary);
    margin-bottom: 0.3rem;
}
.page-heading p { font-size: 0.9rem; color: var(--text-secondary); }

/* ── Upload ── */
[data-testid="stFileUploader"] { width: 100% !important; }

[data-testid="stFileUploaderDropzone"] {
    background: var(--bg-secondary) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 2.5rem !important;
    transition: border-color var(--transition), background var(--transition) !important;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--border-hover) !important;
    background: var(--bg-tertiary) !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] span {
    color: var(--text-secondary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: var(--text-muted) !important;
    font-size: 0.78rem !important;
}

/* ── Image ── */
[data-testid="stImage"] img {
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--border) !important;
}

/* ── Cards ── */
.card {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.75rem;
    transition: border-color var(--transition);
}
.card:hover { border-color: var(--border-hover); }

.card-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.45rem;
}

.card-value {
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.card-value-accent { color: var(--accent); }

/* ── Progress ── */
.prog-wrap { margin-top: 0.75rem; }
.prog-header {
    display: flex;
    justify-content: space-between;
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-bottom: 0.35rem;
    font-family: 'DM Mono', monospace;
}
.prog-track {
    height: 4px;
    background: var(--bg-tertiary);
    border-radius: 999px;
    overflow: hidden;
}
.prog-fill {
    height: 100%;
    border-radius: 999px;
    background: var(--accent);
    transition: width 0.6s cubic-bezier(0.22,1,0.36,1);
}
.prog-fill.medium { background: #e3b341; }
.prog-fill.low    { background: #f47067; }

/* ── Distribution ── */
.dist-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.38rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.dist-row:last-child { border-bottom: none; }
.dist-rank {
    width: 22px;
    font-size: 0.68rem;
    color: var(--text-muted);
    font-family: 'DM Mono', monospace;
    text-align: right;
    flex-shrink: 0;
}
.dist-name {
    flex: 1;
    font-size: 0.85rem;
    color: var(--text-secondary);
    display: flex;
    align-items: center;
    gap: 0.4rem;
    min-width: 0;
}
.dist-name.winner { color: var(--text-primary); font-weight: 500; }
.dist-bar-wrap { width: 90px; flex-shrink: 0; }
.dist-bar {
    height: 3px;
    background: var(--bg-tertiary);
    border-radius: 999px;
    overflow: hidden;
}
.dist-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: var(--border-hover);
}
.dist-bar-fill.winner { background: var(--accent); }
.dist-pct {
    width: 44px;
    text-align: right;
    font-size: 0.75rem;
    font-family: 'DM Mono', monospace;
    color: var(--text-muted);
    flex-shrink: 0;
}
.dist-pct.winner { color: var(--accent); }

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 4rem 1rem;
    color: var(--text-muted);
}
.empty-icon { font-size: 2.4rem; margin-bottom: 0.75rem; opacity: 0.4; }
.empty-text { font-size: 0.88rem; color: var(--text-muted); line-height: 1.6; }

/* ── Misc ── */
.stSpinner > div { border-top-color: var(--accent) !important; }
[data-testid="column"] { padding: 0 0.5rem !important; }
#MainMenu, footer { visibility: hidden; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--bg-tertiary); border-radius: 999px; }
</style>
""", unsafe_allow_html=True)

# ── Model ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    base_path = os.path.dirname(__file__)
    
    st.write("BASE:", base_path)
    st.write("FILES:", os.listdir(base_path))
    
    # 2. Join that folder path with your model filename
    model_path = os.path.join(base_path, 'Fruit_Class2.keras')
    
    # 3. Load using the full path
    return tf.keras.models.load_model(model_path,compile=False)

model = load_model()

CLASS_NAME = ['Apple', 'Banana', 'Avocado', 'Cherry', 'Kiwi',
              'Mango', 'Orange', 'Pineapple', 'Strawberries', 'Watermelon']

FRUIT_ICONS = {
    'Apple': '🍎', 'Banana': '🍌', 'Avocado': '🥑', 'Cherry': '🍒',
    'Kiwi': '🥝', 'Mango': '🥭', 'Orange': '🍊', 'Pineapple': '🍍',
    'Strawberries': '🍓', 'Watermelon': '🍉',
}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">⚡</div>
        <div class="sidebar-logo-name">Fruit-Classifier</div>
    </div>

    <div class="sidebar-item active">
        <span class="sidebar-item-icon">🔍</span> Classifier
    </div>
    <div class="sidebar-item">
        <span class="sidebar-item-icon">📊</span> History
    </div>
    <div class="sidebar-item">
        <span class="sidebar-item-icon">⚙️</span> Settings
    </div>

    <div class="sidebar-section">Model Info</div>
    <div class="model-badge">
        <div class="model-badge-title"><span class="online-dot"></span>EfficientNet B0</div>
        <div class="model-badge-sub">TensorFlow · 10 classes · 224×224</div>
    </div>

    <div class="sidebar-section" style="margin-top:1.75rem;">Supported Fruits</div>
    """, unsafe_allow_html=True)

    for name, icon in FRUIT_ICONS.items():
        st.markdown(f"""
        <div class="sidebar-item" style="padding:0.35rem 0.75rem;">
            <span>{icon}</span>
            <span style="font-size:0.83rem;">{name}</span>
        </div>
        """, unsafe_allow_html=True)

# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-heading">
    <h1>Fruit Classifier</h1>
    <p>Upload a fruit image and get an instant AI prediction with confidence scores.</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1.05, 0.95], gap="large")

with col1:
    uploaded = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        st.image(pil_img, use_container_width=True)
        st.markdown(f"""
        <div style="margin-top:0.5rem;font-size:0.73rem;color:var(--text-muted);
                    font-family:'DM Mono',monospace;">
            {uploaded.name}&nbsp;·&nbsp;{pil_img.width}×{pil_img.height} px
        </div>
        """, unsafe_allow_html=True)

with col2:
    if uploaded:
        with st.spinner("Analyzing image…"):
            img_resized  = pil_img.resize((224, 224))
            img_array    = np.expand_dims(np.array(img_resized), axis=0)
            predictions  = model.predict(img_array, verbose=0)
            score        = predictions[0]

        result     = CLASS_NAME[np.argmax(score)]
        confidence = float(np.max(score)) * 100
        icon       = FRUIT_ICONS.get(result, '🍑')
        tier       = "high" if confidence >= 70 else ("medium" if confidence >= 40 else "low")

        # Detected class
        st.markdown(f"""
        <div class="card">
            <div class="card-label">Detected Class</div>
            <div class="card-value">{icon}&nbsp;{result}</div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence
        st.markdown(f"""
        <div class="card">
            <div class="card-label">Confidence</div>
            <div class="card-value card-value-accent">{confidence:.1f}%</div>
            <div class="prog-wrap">
                <div class="prog-header"><span>0%</span><span>50%</span><span>100%</span></div>
                <div class="prog-track">
                    <div class="prog-fill {tier}" style="width:{confidence:.1f}%"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Top-5 distribution
        top_idx   = np.argsort(score)[::-1][:5]
        rows_html = ""
        for rank, idx in enumerate(top_idx):
            pct  = score[idx] * 100
            is_w = rank == 0
            rows_html += f"""
            <div class="dist-row">
                <div class="dist-rank">#{rank+1}</div>
                <div class="dist-name {'winner' if is_w else ''}">
                    {FRUIT_ICONS.get(CLASS_NAME[idx], '🍑')}&nbsp;{CLASS_NAME[idx]}
                </div>
                <div class="dist-bar-wrap">
                    <div class="dist-bar">
                        <div class="dist-bar-fill {'winner' if is_w else ''}"
                             style="width:{pct:.1f}%"></div>
                    </div>
                </div>
                <div class="dist-pct {'winner' if is_w else ''}">{pct:.1f}%</div>
            </div>"""

        st.markdown(f"""
        <div class="card">
            <div class="card-label">Top 5 Predictions</div>
            <div style="margin-top:0.75rem;">{rows_html}</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🔍</div>
            <div class="empty-text">
                Upload a fruit image on the left<br>to see the AI prediction here.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;padding-top:1.25rem;border-top:1px solid var(--border);
            font-size:0.72rem;color:var(--text-muted);display:flex;
            justify-content:space-between;align-items:center;">
    <span>FruitLens AI · Powered by TensorFlow</span>
    <span>10 Classes · ResNet Architecture</span>
</div>
""", unsafe_allow_html=True)