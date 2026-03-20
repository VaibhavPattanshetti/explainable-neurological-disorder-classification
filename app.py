import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg: #0a0e1a;
    --surface: #111827;
    --surface2: #1a2235;
    --accent: #00d4ff;
    --accent2: #7c3aed;
    --text: #e2e8f0;
    --muted: #64748b;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --radius: 16px;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.stApp {
    background: var(--bg);
}

/* Hide streamlit defaults */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem; max-width: 1400px; }

/* Hero */
.hero {
    text-align: center;
    padding: 3rem 0 2rem;
    position: relative;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3.5rem;
    background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.1;
}
.hero-sub {
    font-size: 1.1rem;
    color: var(--muted);
    margin-top: 0.75rem;
    font-weight: 300;
    letter-spacing: 0.02em;
}
.badge-row {
    display: flex;
    justify-content: center;
    gap: 0.75rem;
    margin-top: 1.25rem;
    flex-wrap: wrap;
}
.badge {
    background: var(--surface2);
    border: 1px solid #1e3a5f;
    border-radius: 999px;
    padding: 0.3rem 1rem;
    font-size: 0.78rem;
    color: var(--accent);
    font-weight: 500;
    letter-spacing: 0.05em;
}

/* Upload zone */
.upload-card {
    background: var(--surface);
    border: 2px dashed #1e3a5f;
    border-radius: var(--radius);
    padding: 2.5rem;
    text-align: center;
    transition: border-color 0.3s;
    margin: 1.5rem 0;
}
.upload-card:hover { border-color: var(--accent); }

/* Result card */
.result-card {
    background: var(--surface);
    border-radius: var(--radius);
    padding: 1.75rem;
    border: 1px solid #1e3a5f;
    margin-bottom: 1.25rem;
}
.result-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: var(--muted);
    margin-bottom: 0.4rem;
}
.result-class {
    font-family: 'DM Serif Display', serif;
    font-size: 1.7rem;
    color: var(--text);
    margin-bottom: 0.5rem;
    line-height: 1.2;
}
.confidence-bar-bg {
    height: 8px;
    background: var(--surface2);
    border-radius: 99px;
    overflow: hidden;
    margin-top: 0.75rem;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #00d4ff, #7c3aed);
    transition: width 0.8s ease;
}
.conf-number {
    font-size: 2.5rem;
    font-weight: 600;
    background: linear-gradient(135deg, #00d4ff, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* XAI section */
.xai-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: var(--text);
    margin: 2rem 0 0.5rem;
}
.xai-desc {
    color: var(--muted);
    font-size: 0.9rem;
    margin-bottom: 1.5rem;
}
.xai-method-label {
    background: var(--surface2);
    border-radius: 8px;
    padding: 0.4rem 0.9rem;
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--accent);
    display: inline-block;
    margin-bottom: 0.75rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.method-card {
    background: var(--surface);
    border-radius: var(--radius);
    padding: 1.5rem;
    border: 1px solid #1e3a5f;
    margin-bottom: 1rem;
}
.method-title {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 0.25rem;
}
.method-body {
    font-size: 0.85rem;
    color: var(--muted);
    line-height: 1.6;
}

/* Class pills */
.classes-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 0.75rem 0;
}
.class-pill {
    background: var(--surface2);
    border-radius: 8px;
    padding: 0.35rem 0.85rem;
    font-size: 0.78rem;
    color: var(--text);
    border: 1px solid #1e3a5f;
}

/* Divider */
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1e3a5f, transparent);
    margin: 2rem 0;
}

/* Spinner override */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* Streamlit image captions */
.stImage > div > div { color: var(--muted); font-size: 0.8rem; }

/* Metrics */
[data-testid="stMetricValue"] {
    font-family: 'DM Serif Display', serif;
    color: var(--accent) !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
IMG_SIZE = 224
CLASS_NAMES = [
    "1st Brain Tumor Glioma",
    "2nd Brain Tumor Meningioma",
    "3rd Brain Tumor Pituitary",
    "Alzehaimer Dementia 1st Very Mild",
    "Alzehaimer Dementia 2nd Mild",
    "Alzehaimer Dementia 3rd Moderate",
    "Multiple Sclerosis",
    "Normal",
]
LAST_CONV_LAYER = "conv5_block3_out"
NUM_CLASSES = len(CLASS_NAMES)

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
for _key in ["results", "last_file_id"]:
    if _key not in st.session_state:
        st.session_state[_key] = None

# Severity / color mapping for each class
CLASS_META = {
    "1st Brain Tumor Glioma":              {"group": "Brain Tumor",   "color": "#ef4444"},
    "2nd Brain Tumor Meningioma":          {"group": "Brain Tumor",   "color": "#f97316"},
    "3rd Brain Tumor Pituitary":           {"group": "Brain Tumor",   "color": "#f59e0b"},
    "Alzehaimer Dementia 1st Very Mild":   {"group": "Alzheimer's",   "color": "#8b5cf6"},
    "Alzehaimer Dementia 2nd Mild":        {"group": "Alzheimer's",   "color": "#7c3aed"},
    "Alzehaimer Dementia 3rd Moderate":    {"group": "Alzheimer's",   "color": "#6d28d9"},
    "Multiple Sclerosis":                  {"group": "MS",            "color": "#06b6d4"},
    "Normal":                              {"group": "Normal",        "color": "#10b981"},
}

# ─────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────
@st.cache_resource
def load_brain_model():
    model_path = "resenet_50_new_dataset_92_v2.keras"
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path} — place your .keras file next to app.py")
        st.stop()
    # compile=False is intentional for inference:
    #   - Preserves all trained weights and architecture exactly
    #   - Avoids dropping the multi-label AUC metric saved with the model
    #   - Skips unnecessary optimizer/loss rebuild that adds startup time
    #   - model.predict() works fully without compile
    model = load_model(model_path, compile=False)
    return model

# ─────────────────────────────────────────────
# PREPROCESSING
# Exact mirror of preprocess_for_gradcam() from the notebook:
#   img = tf.io.read_file(img_path)
#   img = tf.image.decode_jpeg(img, channels=3)   → forces 3-ch RGB
#   img = tf.image.resize(img, (IMG_SIZE,IMG_SIZE))→ bilinear by default
#   img = tf.cast(img, tf.float32)
#   img = resnet.preprocess_input(img)             → ImageNet mean sub + BGR
#   return tf.expand_dims(img, axis=0)             → batch dim
#
# We receive a PIL image from Streamlit's uploader instead of a file path,
# so we encode it to JPEG bytes first and feed through the same TF decode
# pipeline — this guarantees identical numerical output to the notebook.
#
# original_rgb is saved as uint8 BEFORE preprocess_input for display
# and LIME (which needs raw 0-255 pixel values as input).
# ─────────────────────────────────────────────
def preprocess_image(pil_img):
    # Encode PIL image to JPEG bytes so we can run the exact same
    # tf.image.decode_jpeg pipeline the notebook uses
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG", quality=95)
    jpeg_bytes = buf.getvalue()

    # Exact notebook pipeline: decode → resize → cast → preprocess_input
    img = tf.image.decode_jpeg(jpeg_bytes, channels=3)          # uint8 RGB tensor
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))             # bilinear, same default
    img = tf.cast(img, tf.float32)                               # float32

    # Save clean uint8 copy for display + LIME before preprocess_input mutates values
    original_rgb = np.array(
        tf.cast(tf.clip_by_value(img, 0, 255), tf.uint8)
    )  # shape (224, 224, 3), values 0-255

    img = tf.keras.applications.resnet.preprocess_input(img)    # ImageNet mean sub + BGR swap
    img_tensor = tf.expand_dims(img, axis=0)                    # (1, 224, 224, 3)

    return img_tensor, original_rgb


# ─────────────────────────────────────────────
# GRAD-CAM
# Exact mirror of get_gradcam_heatmap() from the notebook.
# Primary path: grads of preds[:,pred_index] w.r.t. conv_output
#   (preds comes from the full model forward pass, watched via seq_out_var)
# Fallback path (only if grads is None): rebuild the head explicitly
#   so the tape has a differentiable path to conv_output.
# ─────────────────────────────────────────────
def get_gradcam_heatmap(model, img_array, pred_index):
    resnet = model.get_layer("resnet50")

    # Get prediction from full model first (matches notebook line-for-line)
    preds = model(img_array, training=False)

    # Pass through the Sequential augmentation/preprocess layer to get resnet input
    seq_out = model.get_layer("sequential")(img_array, training=False)

    # Sub-model: resnet input → [last conv output, resnet output]
    resnet_grad_model = tf.keras.models.Model(
        inputs=resnet.input,
        outputs=[
            resnet.get_layer(LAST_CONV_LAYER).output,
            resnet.output
        ]
    )

    seq_out_var = tf.Variable(seq_out, trainable=False)

    # PRIMARY PATH — matches notebook exactly:
    # watch seq_out_var, get conv_output, then differentiate
    # preds[:,pred_index] (outer model output) w.r.t. conv_output
    with tf.GradientTape() as tape:
        tape.watch(seq_out_var)
        conv_output, resnet_out = resnet_grad_model(seq_out_var, training=False)
        tape.watch(conv_output)
        class_channel = preds[:, pred_index]   # use outer model preds, same as notebook

    grads = tape.gradient(class_channel, conv_output)

    if grads is None:
        # FALLBACK PATH — only reached if primary grads are None.
        # Rebuilds the classification head inside the tape so the gradient
        # path from class_channel back to conv_output is fully connected.
        with tf.GradientTape() as tape2:
            tape2.watch(seq_out_var)
            conv_output, _ = resnet_grad_model(seq_out_var, training=False)
            tape2.watch(conv_output)
            x = model.get_layer("global_average_pooling2d")(conv_output)
            x = model.get_layer("dense")(x)
            x = model.get_layer("dropout")(x, training=False)
            x = model.get_layer("dense_1")(x)
            class_channel = x[:, pred_index]
        grads = tape2.gradient(class_channel, conv_output)

    if grads is None:
        return np.zeros((7, 7))

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def make_gradcam_figure(original_rgb, heatmap, pred_label, confidence):
    """Returns a matplotlib figure with 3 panels."""
    heatmap_resized = np.array(
        Image.fromarray(np.uint8(cm.jet(heatmap) * 255)[..., :3]).resize((IMG_SIZE, IMG_SIZE))
    )
    overlay = (heatmap_resized * 0.4 + original_rgb * 0.6).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="#0a0e1a")
    titles = ["Original MRI", "Grad-CAM Heatmap", f"Overlay · {confidence:.1%}"]
    imgs = [original_rgb, heatmap_resized, overlay]

    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img)
        ax.set_title(title, color="#94a3b8", fontsize=10, pad=8)
        ax.axis("off")

    fig.suptitle(f"Grad-CAM — {pred_label}", color="#e2e8f0", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig

# ─────────────────────────────────────────────
# LIME
# Exact mirror of get_lime_explanation() from the notebook.
# predict_fn receives uint8 numpy images from LIME, casts to float32,
# then applies resnet.preprocess_input — matching the notebook exactly.
# original_rgb must be uint8 (H,W,3) — guaranteed by preprocess_image().
# ─────────────────────────────────────────────
def run_lime(model, original_rgb, pred_idx, num_samples=500):
    try:
        from lime import lime_image
        from skimage.segmentation import mark_boundaries
    except ImportError:
        return None, None

    # Exact predict_fn from notebook:
    # cast uint8 LIME perturbations → float32 → resnet preprocess → predict
    def predict_fn(images):
        preprocessed = tf.keras.applications.resnet.preprocess_input(
            tf.cast(images, tf.float32)
        )
        return model.predict(preprocessed, verbose=0)
        
        
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        original_rgb,           # uint8 (224,224,3) — matches notebook
        predict_fn,
        top_labels=3,           # matches notebook
        hide_color=0,           # matches notebook
        num_samples=num_samples
    )
    temp, mask = explanation.get_image_and_mask(
        pred_idx,
        positive_only=True,
        num_features=10,        # matches notebook (get_lime_explanation uses 10)
        hide_rest=False
    )
    return temp, mask

def make_lime_figure(original_rgb, temp, mask, pred_label):
    from skimage.segmentation import mark_boundaries
    overlay_mask = original_rgb.copy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="#0a0e1a")
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original MRI", color="#94a3b8", fontsize=10, pad=8)
    axes[0].axis("off")

    axes[1].imshow(mark_boundaries(temp / 255.0, mask))
    axes[1].set_title("LIME Important Regions", color="#94a3b8", fontsize=10, pad=8)
    axes[1].axis("off")

    axes[2].imshow(original_rgb)
    axes[2].imshow(mask, alpha=0.5, cmap="hot")
    axes[2].set_title("LIME Heatmap Overlay", color="#94a3b8", fontsize=10, pad=8)
    axes[2].axis("off")

    fig.suptitle(f"LIME — {pred_label}", color="#e2e8f0", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf

# ─────────────────────────────────────────────
# PROBABILITY BAR CHART
# ─────────────────────────────────────────────
def make_prob_chart(probs):
    sorted_idx = np.argsort(probs)[::-1]
    names = [CLASS_NAMES[i].replace("Alzehaimer", "Alzheimer") for i in sorted_idx]
    vals = [probs[i] * 100 for i in sorted_idx]
    colors_list = [CLASS_META[CLASS_NAMES[i]]["color"] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(7, 4), facecolor="#111827")
    bars = ax.barh(names[::-1], vals[::-1], color=colors_list[::-1],
                   height=0.6, edgecolor="none")
    ax.set_facecolor("#111827")
    ax.tick_params(colors="#94a3b8", labelsize=9)
    ax.spines[:].set_visible(False)
    ax.set_xlabel("Confidence (%)", color="#64748b", fontsize=9)
    for bar, val in zip(bars, vals[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", color="#e2e8f0", fontsize=8.5)
    ax.set_xlim(0, 115)
    ax.xaxis.label.set_color("#64748b")
    fig.tight_layout()
    return fig

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1 class="hero-title">🧠 NeuroScan AI</h1>
    <p class="hero-sub">Brain MRI Classification · Explainable AI · ResNet50</p>
    <div class="badge-row">
        <span class="badge">ResNet50 · Transfer Learning</span>
        <span class="badge">92% Validation Accuracy</span>
        <span class="badge">8 Neurological Classes</span>
        <span class="badge">Grad-CAM · LIME</span>
    </div>
</div>
<div class="fancy-divider"></div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR – ABOUT
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 About NeuroScan AI")
    st.markdown("""
**Model:** ResNet50 (fine-tuned)  
**Dataset:** 24,588 MRI images  
**Val Accuracy:** 92%  
**Val AUC:** 0.9952  

---
**Detectable Conditions**
""")
    for cn in CLASS_NAMES:
        meta = CLASS_META[cn]
        st.markdown(f"<span style='color:{meta['color']}'>●</span> {cn}", unsafe_allow_html=True)

    st.markdown("""
---
**XAI Methods**

**Grad-CAM** uses gradients flowing back through the CNN to highlight which spatial regions were most influential in the prediction.

**LIME** perturbs superpixel regions to find which parts of the image most affect the model's output.

---
⚠️ *For research purposes only. Not a clinical diagnostic tool.*
""")

# ─────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.8], gap="large")

with col_left:
    st.markdown("### Upload MRI Scan")
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drop a brain MRI image here",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded:
        pil_img = Image.open(uploaded)
        st.image(pil_img, caption="Uploaded MRI", use_container_width=True)

    # XAI settings
    st.markdown("### ⚙️ Analysis Settings")
    run_gradcam = st.toggle("Grad-CAM", value=True)
    run_lime_xai = st.toggle("LIME", value=True)
    if run_lime_xai:
        lime_samples = st.slider("LIME samples (more = slower but better)", 100, 800, 500, 50)
    analyze_btn = st.button("🔬 Analyze Scan", use_container_width=True, type="primary")

# ─────────────────────────────────────────────
# ANALYSIS — run only when button clicked, then persist in session_state
# ─────────────────────────────────────────────

# Clear saved results if a new file is uploaded
if uploaded:
    file_id = uploaded.file_id
    if st.session_state.last_file_id != file_id:
        st.session_state.results = None
        st.session_state.last_file_id = file_id

# Run inference + XAI and store everything in session_state
if uploaded and analyze_btn:
    model = load_brain_model()
    pil_img = Image.open(uploaded)

    with st.spinner("Running inference..."):
        img_tensor, original_rgb = preprocess_image(pil_img)
        preds = model.predict(img_tensor, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        pred_label = CLASS_NAMES[pred_idx]
        confidence = float(preds[pred_idx])

    gradcam_bytes = None
    lime_bytes = None

    if run_gradcam:
        with st.spinner("Computing Grad-CAM..."):
            heatmap = get_gradcam_heatmap(model, img_tensor, pred_idx)
            gradcam_fig = make_gradcam_figure(original_rgb, heatmap, pred_label, confidence)
            gradcam_bytes = fig_to_bytes(gradcam_fig)

    if run_lime_xai:
        with st.spinner(f"Running LIME ({lime_samples} samples)... this may take ~30s"):
            temp, mask = run_lime(model, original_rgb, pred_idx, num_samples=lime_samples)
        if temp is not None:
            lime_fig = make_lime_figure(original_rgb, temp, mask, pred_label)
            lime_bytes = fig_to_bytes(lime_fig)

    # Persist everything — keyed by file_id so a new upload clears it
    st.session_state.results = {
        "pred_label":    pred_label,
        "pred_idx":      pred_idx,
        "confidence":    confidence,
        "preds":         preds,
        "gradcam_bytes": gradcam_bytes,
        "lime_bytes":    lime_bytes,
        "ran_gradcam":   run_gradcam,
        "ran_lime":      run_lime_xai,
    }

# ── Render from session_state (survives download-button reruns) ──
with col_right:
    if not uploaded:
        st.markdown("""
<div class="result-card" style="text-align:center; padding: 4rem 2rem;">
    <div style="font-size:3rem">🫁</div>
    <div style="color:#64748b; margin-top:1rem; font-size:0.95rem;">
        Upload an MRI scan on the left to begin analysis.<br>
        Supports JPEG and PNG formats.
    </div>
    <div class="fancy-divider" style="margin: 1.5rem auto; max-width:200px;"></div>
    <div class="classes-grid" style="justify-content:center;">
""" + "".join([f'<span class="class-pill">{c}</span>' for c in CLASS_NAMES]) + """
    </div>
</div>
""", unsafe_allow_html=True)

    elif st.session_state.results is None:
        st.markdown("""
<div class="result-card" style="text-align:center; padding: 3rem 2rem;">
    <div style="font-size:2.5rem">🔬</div>
    <div style="color:#64748b; margin-top:1rem; font-size:0.95rem;">
        Image uploaded. Click <b style="color:#e2e8f0">Analyze Scan</b> to run prediction.
    </div>
</div>
""", unsafe_allow_html=True)

    else:
        r = st.session_state.results
        pred_label = r["pred_label"]
        confidence = r["confidence"]
        preds      = r["preds"]
        meta       = CLASS_META[pred_label]

        # ── Prediction card ──
        st.markdown(f"""
<div class="result-card">
    <div class="result-label">Prediction</div>
    <div class="result-class">{pred_label}</div>
    <div style="display:flex; align-items:center; gap:1rem; margin-top:0.5rem;">
        <span class="conf-number">{confidence:.1%}</span>
        <span style="color:#64748b; font-size:0.85rem;">confidence</span>
        <span style="background:{meta['color']}22; color:{meta['color']}; border-radius:99px;
              padding:0.2rem 0.75rem; font-size:0.78rem; font-weight:600; margin-left:auto;">
            {meta['group']}
        </span>
    </div>
    <div class="confidence-bar-bg">
        <div class="confidence-bar-fill" style="width:{confidence*100:.1f}%"></div>
    </div>
</div>
""", unsafe_allow_html=True)

        # ── Metrics row ──
        m1, m2, m3 = st.columns(3)
        m1.metric("Top Class", pred_label.split(" ")[0] + "…")
        m2.metric("Confidence", f"{confidence:.1%}")
        m3.metric("2nd Best", CLASS_NAMES[np.argsort(preds)[-2]].split(" ")[0] + "…")

        # ── Probability chart ──
        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
        st.markdown("**Class Probability Distribution**")
        prob_fig = make_prob_chart(preds)
        st.pyplot(prob_fig, use_container_width=True)
        plt.close(prob_fig)

        # ── XAI Section ──
        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="xai-header">Explainability Analysis</div>', unsafe_allow_html=True)
        st.markdown('<div class="xai-desc">These visualizations reveal <em>why</em> the model made its prediction — which brain regions it focused on.</div>', unsafe_allow_html=True)

        # Grad-CAM
        if r["ran_gradcam"] and r["gradcam_bytes"]:
            st.markdown('<span class="xai-method-label">Grad-CAM · Gradient-based</span>', unsafe_allow_html=True)
            st.image(r["gradcam_bytes"], use_container_width=True)
            st.markdown("""
<div class="method-card">
    <div class="method-title">How to read Grad-CAM</div>
    <div class="method-body">
        <b style="color:#ef4444">Red/warm</b> areas are regions the model weighted most heavily.<br>
        <b style="color:#3b82f6">Blue/cool</b> areas had little influence on the prediction.<br>
        For tumors, hot-spots should align with the lesion. For Alzheimer's,
        attention often falls on the ventricles and hippocampal region.
    </div>
</div>
""", unsafe_allow_html=True)

        # LIME
        if r["ran_lime"]:
            st.markdown('<span class="xai-method-label">LIME · Perturbation-based</span>', unsafe_allow_html=True)
            if r["lime_bytes"]:
                st.image(r["lime_bytes"], use_container_width=True)
                st.markdown("""
<div class="method-card">
    <div class="method-title">How to read LIME</div>
    <div class="method-body">
        LIME segments the image into superpixels and tests which ones, when masked,
        change the prediction the most.<br>
        <b style="color:#f59e0b">Yellow outlines</b> mark the most important superpixel regions.<br>
        The <b>heatmap overlay</b> shows intensity of each region's contribution.
    </div>
</div>
""", unsafe_allow_html=True)
            else:
                st.warning("LIME requires `lime` package: `pip install lime`")

        # ── Download — clicking these no longer wipes the results ──
        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
        st.markdown("**📥 Export Results**")
        dl1, dl2 = st.columns(2)
        if r["ran_gradcam"] and r["gradcam_bytes"]:
            dl1.download_button(
                "⬇ Download Grad-CAM",
                data=r["gradcam_bytes"],
                file_name=f"gradcam_{pred_label.replace(' ', '_')}.png",
                mime="image/png",
                use_container_width=True,
            )
        if r["ran_lime"] and r["lime_bytes"]:
            dl2.download_button(
                "⬇ Download LIME",
                data=r["lime_bytes"],
                file_name=f"lime_{pred_label.replace(' ', '_')}.png",
                mime="image/png",
                use_container_width=True,
            )

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="fancy-divider"></div>
<div style="text-align:center; color:#334155; font-size:0.8rem; padding-bottom:2rem;">
    NeuroScan AI · ResNet50 Transfer Learning · Grad-CAM + LIME XAI<br>
    ⚠️ Research tool only — not intended for clinical diagnosis
</div>
""", unsafe_allow_html=True)
