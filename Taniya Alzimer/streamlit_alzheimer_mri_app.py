import io
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image, ImageOps
from torchvision import transforms

try:
    import nibabel as nib
except ImportError:
    nib = None


# ---------------------------------------------------------
# App configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Alzheimer's MRI Classifier",
    page_icon=":material/neurology:",
    layout="wide",
)

# Path to the trained model. Change this if you move the .pth file.
CHECKPOINT_PATH = Path(
    "/Users/abhishekmishra/Desktop/Data/Model Deploymeent/Taniya Alzimer/best_hybrid_cnn_vit.pth"
)

# Observed on the project's scan-level test split.
TEST_ACCURACY = 0.6064

CLASS_NAMES = {0: "CN", 1: "MCI", 2: "AD"}
CLASS_INFO = {
    "CN": {
        "title": "Cognitively Normal",
        "color": "#10B981",
        "about": (
            "Participants without cognitive impairment. Brain structure is "
            "within the typical range for age in the training data."
        ),
    },
    "MCI": {
        "title": "Mild Cognitive Impairment",
        "color": "#F59E0B",
        "about": (
            "Memory or thinking decline greater than expected for age, while "
            "daily activities are largely intact. Some people with MCI later "
            "progress to Alzheimer's disease."
        ),
    },
    "AD": {
        "title": "Alzheimer's Disease",
        "color": "#EF4444",
        "about": (
            "Alzheimer's dementia group. On MRI it is commonly associated with "
            "hippocampal and cortical atrophy and enlarged ventricles."
        ),
    },
}

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem;}
    .hero {
        background: linear-gradient(120deg, #312E81 0%, #4F46E5 55%, #7C3AED 100%);
        border-radius: 18px;
        padding: 1.8rem 2rem;
        color: #FFFFFF;
        margin-bottom: 1.2rem;
    }
    .hero h1 {color: #FFFFFF; font-size: 2.1rem; margin: 0 0 .4rem 0; padding: 0;}
    .hero p {color: rgba(255,255,255,.88); font-size: 1.02rem; margin: 0;}
    .hero .tags {margin-top: .9rem;}
    .hero .tag {
        display: inline-block; background: rgba(255,255,255,.16);
        border: 1px solid rgba(255,255,255,.25); border-radius: 999px;
        padding: .2rem .75rem; margin: 0 .4rem .3rem 0; font-size: .82rem;
    }
    .card {
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        background: rgba(128,128,128,.06);
        height: 100%;
    }
    .card h4 {margin: 0 0 .35rem 0; padding: 0;}
    .card p {margin: 0; opacity: .85; font-size: .93rem;}
    .result-card {
        border-radius: 16px;
        padding: 1.4rem 1.5rem;
        color: #FFFFFF;
    }
    .result-card .label {font-size: .85rem; opacity: .9; text-transform: uppercase; letter-spacing: .08em;}
    .result-card .cls {font-size: 2.6rem; font-weight: 800; line-height: 1.1; margin: .2rem 0;}
    .result-card .sub {font-size: 1.05rem; opacity: .95;}
    .result-card .conf {font-size: 1.6rem; font-weight: 700; margin-top: .6rem;}
    .dot {
        display: inline-block; width: .7rem; height: .7rem; border-radius: 50%;
        margin-right: .5rem; vertical-align: middle;
    }
    .section {
        border-left: 5px solid #4F46E5;
        padding: .2rem 0 .2rem 1rem;
        margin: 2.6rem 0 1.1rem 0;
        scroll-margin-top: 4rem;
    }
    .section h2 {margin: 0; padding: 0; font-size: 1.55rem;}
    .section p {margin: .2rem 0 0 0; opacity: .75;}
    .jump-nav {margin: .2rem 0 .4rem 0;}
    .jump-nav a {
        display: inline-block; text-decoration: none; font-size: .85rem; font-weight: 600;
        border: 1px solid rgba(79,70,229,.45); color: #6366F1; border-radius: 999px;
        padding: .25rem .85rem; margin: 0 .4rem .4rem 0;
    }
    .jump-nav a:hover {background: rgba(79,70,229,.1);}
    .pill {
        display: inline-block; border-radius: 999px; padding: .15rem .7rem;
        font-size: .8rem; font-weight: 600; color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Model architecture must match the training notebook
# ---------------------------------------------------------
class HybridCNNViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = timm.create_model(
            "efficientnet_b4", pretrained=False, num_classes=0
        )
        self.vit = timm.create_model(
            "vit_base_patch16_224", pretrained=False, num_classes=0
        )
        self.classifier = nn.Sequential(
            nn.Linear(1792 + 768, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 3),
        )

    def forward(self, x):
        cnn_features = self.cnn(x)
        vit_features = self.vit(x)
        fused_features = torch.cat([cnn_features, vit_features], dim=1)
        return self.classifier(fused_features)


@st.cache_resource
def load_model_from_path(checkpoint_path: str, modified_time: float):
    # modified_time is only part of the cache key, so a retrained checkpoint reloads.
    return load_model(checkpoint_path)


def load_model(checkpoint_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridCNNViT()

    # Supports both the project's saved checkpoint dictionary and a raw state_dict.
    checkpoint = torch.load(
        checkpoint_file, map_location=device, weights_only=False
    )
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        saved_epoch = checkpoint.get("epoch")
        saved_val_accuracy = checkpoint.get("val_accuracy")
    else:
        state_dict = checkpoint
        saved_epoch = None
        saved_val_accuracy = None

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, device, saved_epoch, saved_val_accuracy


# Match the training notebook's evaluation transform:
# resize to 224x224, grayscale -> 3 channels, ImageNet normalization.
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def normalize_mri_slice(array: np.ndarray) -> Image.Image:
    """Use the notebook's percentile-based normalization for a 2D MRI slice."""
    array = np.asarray(array, dtype=np.float32)
    nonzero = array[array > 0]

    if nonzero.size == 0:
        raise ValueError("The selected MRI slice contains no non-zero pixels.")

    low, high = np.percentile(nonzero, [1, 99])
    if high <= low:
        low = float(np.min(nonzero))
        high = float(np.max(nonzero))
    if high <= low:
        raise ValueError("The MRI slice has no usable intensity range.")

    scaled = np.clip((array - low) / (high - low), 0, 1)
    scaled[array <= 0] = 0
    image_array = (scaled * 255).astype(np.uint8)
    return Image.fromarray(image_array)


@st.cache_data(show_spinner=False)
def load_nifti(data: bytes):
    if nib is None:
        raise ImportError("Install nibabel to read NIfTI files: pip install nibabel")
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as temp:
        temp.write(data)
        temp_path = temp.name
    try:
        nifti = nib.load(temp_path)
        volume = np.asarray(nifti.get_fdata(), dtype=np.float32)
        zooms = tuple(float(z) for z in nifti.header.get_zooms()[:3])
    finally:
        Path(temp_path).unlink(missing_ok=True)

    if volume.ndim > 3:
        volume = volume[..., 0]
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI MRI; got shape {volume.shape}.")
    return volume, zooms


def read_2d_image(data: bytes):
    image = Image.open(io.BytesIO(data))
    image = ImageOps.exif_transpose(image)
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    # Colour images have noticeably different R/G/B channels; MRI slices do not.
    is_colour = float(np.abs(rgb[..., 0] - rgb[..., 1]).mean() + np.abs(rgb[..., 1] - rgb[..., 2]).mean()) > 6
    return image.convert("L"), is_colour


def take_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    return np.take(volume, index, axis=axis)


# ---------------------------------------------------------
# Inference and explainability
# ---------------------------------------------------------
def predict_batch(model, device, images):
    batch = torch.stack([eval_transform(image) for image in images]).to(device)
    with torch.no_grad():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            logits = model(batch)
    return torch.softmax(logits.float(), dim=1).cpu().numpy()


@st.cache_data(show_spinner=False)
def cached_predict(_model, _device, image_arrays: list):
    images = [Image.fromarray(array) for array in image_arrays]
    return predict_batch(_model, _device, images)


@st.cache_data(show_spinner=False)
def grad_cam(_model, _device, image_array: np.ndarray, class_index: int):
    """Grad-CAM on the EfficientNet branch's final feature map."""
    store = {}

    def forward_hook(module, inputs, output):
        store["activations"] = output
        output.register_hook(lambda grad: store.__setitem__("gradients", grad))

    handle = _model.cnn.bn2.register_forward_hook(forward_hook)
    try:
        input_tensor = eval_transform(Image.fromarray(image_array))
        input_tensor = input_tensor.unsqueeze(0).to(_device)
        with torch.enable_grad():
            logits = _model(input_tensor)
            _model.zero_grad(set_to_none=True)
            logits[0, class_index].backward()
    finally:
        handle.remove()
        _model.zero_grad(set_to_none=True)

    activations = store["activations"].detach()
    gradients = store["gradients"].detach()
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam.cpu().numpy()


def overlay_heatmap(gray: np.ndarray, cam: np.ndarray, opacity: float) -> np.ndarray:
    heat = matplotlib.colormaps["jet"](cam)[..., :3] * 255
    base = np.stack([gray] * 3, axis=-1).astype(np.float32)
    blended = (1 - opacity) * base + opacity * heat
    return blended.clip(0, 255).astype(np.uint8)


def certainty_summary(probabilities: np.ndarray):
    ordered = np.sort(probabilities)[::-1]
    margin = float(ordered[0] - ordered[1])
    if ordered[0] >= 0.7 and margin >= 0.3:
        level = "High"
    elif ordered[0] >= 0.5 and margin >= 0.15:
        level = "Moderate"
    else:
        level = "Low"
    return margin, level


# ---------------------------------------------------------
# Charts
# ---------------------------------------------------------
def probability_chart(probabilities: np.ndarray):
    names = [CLASS_NAMES[i] for i in range(3)]
    fig = go.Figure(go.Bar(
        x=probabilities * 100,
        y=[f"{n} · {CLASS_INFO[n]['title']}" for n in names],
        orientation="h",
        marker_color=[CLASS_INFO[n]["color"] for n in names],
        text=[f"{p * 100:.1f}%" for p in probabilities],
        textposition="outside",
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        height=230,
        margin=dict(l=10, r=40, t=10, b=10),
        xaxis=dict(range=[0, 110], title="Model probability (%)", showgrid=True),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def stage_gauge(probabilities: np.ndarray):
    stage_score = float(probabilities[1] * 1 + probabilities[2] * 2)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=stage_score,
        number={"valueformat": ".2f"},
        gauge={
            "axis": {"range": [0, 2], "tickvals": [0, 1, 2], "ticktext": ["CN", "MCI", "AD"]},
            "bar": {"color": "#4F46E5", "thickness": 0.3},
            "steps": [
                {"range": [0, 0.67], "color": "rgba(16,185,129,.35)"},
                {"range": [0.67, 1.33], "color": "rgba(245,158,11,.35)"},
                {"range": [1.33, 2], "color": "rgba(239,68,68,.35)"},
            ],
        },
    ))
    fig.update_layout(
        height=230,
        margin=dict(l=25, r=25, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig, stage_score


def intensity_histogram(gray: np.ndarray):
    foreground = gray[gray > 10]
    fig = go.Figure(go.Histogram(
        x=foreground,
        nbinsx=64,
        marker_color="#6366F1",
        hovertemplate="Intensity %{x}: %{y} pixels<extra></extra>",
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Pixel intensity (background removed)",
        yaxis_title="Pixels",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def slice_trend_chart(indices, slice_probabilities, selected_index):
    fig = go.Figure()
    for class_index, name in CLASS_NAMES.items():
        fig.add_trace(go.Scatter(
            x=indices,
            y=slice_probabilities[:, class_index] * 100,
            mode="lines+markers",
            name=name,
            line=dict(color=CLASS_INFO[name]["color"], width=3),
            hovertemplate=f"{name} · slice %{{x}}: %{{y:.1f}}%<extra></extra>",
        ))
    fig.add_vline(x=selected_index, line_dash="dash", line_color="#6366F1",
                  annotation_text="selected slice")
    fig.update_layout(
        height=330,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Slice index",
        yaxis=dict(title="Model probability (%)", range=[0, 100]),
        legend=dict(orientation="h", y=1.12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def class_dot(name: str) -> str:
    return f'<span class="dot" style="background:{CLASS_INFO[name]['color']}"></span>'


def section(anchor: str, title: str, subtitle: str):
    st.markdown(
        f'<div class="section" id="{anchor}"><h2>{title}</h2><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def card(title: str, body: str):
    st.markdown(f'<div class="card"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>Alzheimer's MRI Classifier</h1>
      <p>A hybrid EfficientNet-B4 + Vision Transformer model that sorts a brain MRI
      slice into Cognitively Normal, Mild Cognitive Impairment or Alzheimer's Disease.</p>
      <div class="tags">
        <span class="tag">Hybrid CNN–ViT</span>
        <span class="tag">3 classes · CN / MCI / AD</span>
        <span class="tag">PNG · JPG · NIfTI</span>
        <span class="tag">Grad-CAM attention maps</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not CHECKPOINT_PATH.exists():
    st.error(f"Model file not found: `{CHECKPOINT_PATH}`")
    st.info("Update `CHECKPOINT_PATH` at the top of the app to point to your .pth file.")
    st.stop()

try:
    with st.spinner("Loading the Hybrid CNN–ViT model…"):
        model, device, saved_epoch, saved_val_accuracy = load_model_from_path(
            str(CHECKPOINT_PATH), CHECKPOINT_PATH.stat().st_mtime
        )
except Exception as exc:
    st.error(f"Could not load the checkpoint: {exc}")
    st.info(
        "Confirm that CHECKPOINT_PATH points to the Hybrid CNN–ViT checkpoint, not an "
        "EfficientNet-B4, ResNet50, or ViT-only checkpoint."
    )
    st.stop()

parameter_count = sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.header("Upload MRI")
    mri_upload = st.file_uploader(
        "2D slice image or 3D NIfTI volume",
        type=["png", "jpg", "jpeg", "nii", "gz"],
        help="Supported: PNG/JPG/JPEG, .nii, .nii.gz",
    )

    st.header("Analysis options")
    show_cam = st.toggle("Attention map (Grad-CAM)", value=True,
                         help="Highlights the regions that pushed the CNN branch toward its prediction.")
    cam_opacity = st.slider("Heatmap opacity", 0.0, 1.0, 0.45, 0.05, disabled=not show_cam)

    volume_controls = st.container()

    st.divider()
    st.success(f"Model ready on **{str(device).upper()}**")
    st.caption(f"Checkpoint: `{CHECKPOINT_PATH.name}`")
    st.caption(f"Parameters: {parameter_count / 1e6:.1f} M")
    if saved_epoch is not None:
        st.caption(f"Saved epoch: {saved_epoch}")
    if saved_val_accuracy is not None:
        st.caption(f"Validation accuracy: {saved_val_accuracy * 100:.2f}%")
    st.caption(f"Test accuracy: {TEST_ACCURACY * 100:.2f}%")


# ---------------------------------------------------------
# Landing page
# ---------------------------------------------------------
if mri_upload is None:
    st.subheader("How it works")
    step_columns = st.columns(3)
    with step_columns[0]:
        card("1 · Upload", "Drop a 2D MRI slice (PNG/JPG) or a full 3D NIfTI volume "
             "(.nii / .nii.gz) in the sidebar.")
    with step_columns[1]:
        card("2 · Preprocess", "The slice is intensity-normalized, resized to 224 × 224 and "
             "fed to both the EfficientNet-B4 and ViT-B/16 branches.")
    with step_columns[2]:
        card("3 · Interpret", "See class probabilities, model certainty, an attention heatmap "
             "and, for volumes, how predictions change across slices.")

    st.subheader("What the classes mean")
    class_columns = st.columns(3)
    for column, name in zip(class_columns, CLASS_INFO):
        info = CLASS_INFO[name]
        with column:
            card(f'{class_dot(name)}{name} · {info["title"]}', info["about"])

    st.subheader("Model at a glance")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Test accuracy", f"{TEST_ACCURACY * 100:.1f}%",
                             f"+{(TEST_ACCURACY - 1 / 3) * 100:.1f} pts vs chance")
    if saved_val_accuracy is not None:
        metric_columns[1].metric("Validation accuracy", f"{saved_val_accuracy * 100:.1f}%")
    metric_columns[2].metric("Parameters", f"{parameter_count / 1e6:.1f} M")
    metric_columns[3].metric("Input size", "224 × 224")

    st.warning(
        "Research prototype only. This model is not clinically validated and must "
        "not be used to diagnose or guide treatment. Predictions may be wrong."
    )
    st.info("Upload an MRI in the sidebar to begin.")
    st.stop()


# ---------------------------------------------------------
# Read the uploaded MRI
# ---------------------------------------------------------
filename = mri_upload.name.lower()
is_volume = filename.endswith(".nii") or filename.endswith(".nii.gz")
volume = None
is_colour = False

try:
    if is_volume:
        volume, voxel_sizes = load_nifti(mri_upload.getvalue())
        axis_labels = {2: "Axis 2 (training plane)", 1: "Axis 1", 0: "Axis 0"}
        with volume_controls:
            st.header("Volume options")
            axis = st.selectbox("Slicing axis", [2, 1, 0], format_func=axis_labels.get,
                                help="The model was trained on slices taken along axis 2.")
            depth = volume.shape[axis]
            slice_index = st.slider("Slice", 0, depth - 1, depth // 2,
                                    help="Training used the middle slice.")
            run_sweep = st.toggle("Analyze neighbouring slices", value=True)
            sweep_span = st.slider("Slices each side", 2, 20, 8, disabled=not run_sweep)
            sweep_step = st.select_slider("Slice step", [1, 2, 3, 4], value=2, disabled=not run_sweep)
        display_image = normalize_mri_slice(take_slice(volume, axis, slice_index))
        input_note = f"NIfTI volume · slice {slice_index} of {depth} along axis {axis}"
    else:
        display_image, is_colour = read_2d_image(mri_upload.getvalue())
        input_note = "2D image · converted to grayscale"
except Exception as exc:
    st.error(f"Could not process the uploaded MRI: {exc}")
    st.stop()

original_gray = np.asarray(display_image, dtype=np.uint8)
model_view = np.asarray(display_image.resize((224, 224), Image.BILINEAR), dtype=np.uint8)

with st.spinner("Running the model…"):
    probabilities = cached_predict(model, device, [original_gray])[0]

predicted_index = int(np.argmax(probabilities))
predicted_class = CLASS_NAMES[predicted_index]
confidence = float(probabilities[predicted_index])
runner_up_index = int(np.argsort(probabilities)[-2])
margin, certainty_level = certainty_summary(probabilities)
info = CLASS_INFO[predicted_class]


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------
sections = [("prediction", "Prediction"), ("attention-map", "Attention map"),
            ("image-analysis", "Image analysis")]
if is_volume:
    sections.append(("volume-insights", "Volume insights"))
sections.append(("about-the-model", "About the model"))
st.markdown(
    '<div class="jump-nav">'
    + "".join(f'<a href="#{anchor}" target="_self">{title}</a>' for anchor, title in sections)
    + "</div>",
    unsafe_allow_html=True,
)

section("prediction", "Prediction", f"Model output for {mri_upload.name}")
with st.container():
    image_column, result_column = st.columns([1, 1.35], gap="large")

    with image_column:
        st.image(display_image, caption=input_note, use_container_width=True)

    with result_column:
        st.markdown(
            f"""
            <div class="result-card" style="background: linear-gradient(135deg, {info['color']}, {info['color']}CC);">
              <div class="label">Model prediction</div>
              <div class="cls">{predicted_class}</div>
              <div class="sub">{info['title']}</div>
              <div class="conf">{confidence * 100:.1f}% confidence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")

        level_colors = {"High": "#10B981", "Moderate": "#F59E0B", "Low": "#EF4444"}
        metric_columns = st.columns(3)
        metric_columns[0].metric("Certainty", certainty_level)
        metric_columns[1].metric("Lead over runner-up", f"{margin * 100:.1f} pts",
                                 help="Top probability minus the second-highest probability.")
        metric_columns[2].metric("Runner-up", CLASS_NAMES[runner_up_index],
                                 f"{probabilities[runner_up_index] * 100:.1f}%", delta_color="off")

        st.markdown(
            f'<span class="pill" style="background:{level_colors[certainty_level]}">'
            f'{certainty_level} certainty</span>&nbsp; Runner-up: '
            f'<b>{CLASS_NAMES[runner_up_index]}</b> at '
            f'{probabilities[runner_up_index] * 100:.1f}%',
            unsafe_allow_html=True,
        )
        if certainty_level == "Low":
            st.info("The model is split between classes for this input. Treat this "
                    "prediction as especially uncertain.")

    st.write("")
    chart_column, gauge_column = st.columns([1.4, 1], gap="large")
    with chart_column:
        st.markdown("##### Class probabilities")
        st.plotly_chart(probability_chart(probabilities), use_container_width=True,
                        config={"displayModeBar": False})
    with gauge_column:
        gauge, stage_score = stage_gauge(probabilities)
        st.markdown("##### Position on the CN → MCI → AD spectrum")
        st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})
        st.caption("Probability-weighted position (0 = CN, 1 = MCI, 2 = AD). "
                   "A summary of the model output, not a clinical score.")

    with st.expander(f"About {predicted_class} · {info['title']}"):
        st.write(info["about"])

    report = "\n".join([
        "Alzheimer's MRI Classifier - prediction report",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
        f"File: {mri_upload.name}",
        f"Input: {input_note}",
        "",
        f"Predicted class: {predicted_class} ({info['title']})",
        f"Confidence: {confidence * 100:.2f}%",
        f"Certainty: {certainty_level} (lead over runner-up {margin * 100:.1f} pts)",
        f"Spectrum position: {stage_score:.2f} (0 = CN, 2 = AD)",
        "",
        *[f"{CLASS_NAMES[i]}: {probabilities[i] * 100:.2f}%" for i in range(3)],
        "",
        f"Model: Hybrid EfficientNet-B4 + ViT-B/16, test accuracy {TEST_ACCURACY * 100:.2f}%",
        "Research prototype only - not a diagnosis.",
    ])
    st.download_button("Download report", report,
                       file_name=f"mri_prediction_{Path(mri_upload.name).stem}.txt")

    st.caption(
        "Scores are model softmax probabilities, not calibrated clinical probabilities "
        "or a diagnosis."
    )

section("attention-map", "Attention map",
        "Which parts of the image drove the prediction")
with st.container():
    if not show_cam:
        st.info("Turn on **Attention map (Grad-CAM)** in the sidebar to see this view.")
    else:
        cam_class = st.radio(
            "Explain the score for",
            list(CLASS_NAMES.values()),
            index=predicted_index,
            horizontal=True,
            format_func=lambda n: f"{n} ({probabilities[list(CLASS_NAMES.values()).index(n)] * 100:.1f}%)",
        )
        cam_index = list(CLASS_NAMES.values()).index(cam_class)
        with st.spinner("Computing attention map…"):
            cam = grad_cam(model, device, original_gray, cam_index)

        columns = st.columns(3)
        columns[0].image(model_view, caption="Model input (224 × 224)", use_container_width=True)
        columns[1].image(overlay_heatmap(model_view, cam, cam_opacity),
                         caption=f"Grad-CAM overlay for {cam_class}", use_container_width=True)
        columns[2].image(overlay_heatmap(model_view, cam, 1.0),
                         caption="Heatmap only", use_container_width=True)

        focus_share = float((cam > 0.5).mean() * 100)
        brain_mask = model_view > 10
        inside_brain = float(cam[brain_mask].sum() / max(cam.sum(), 1e-8) * 100)
        metric_columns = st.columns(2)
        metric_columns[0].metric("Image area with strong attention", f"{focus_share:.1f}%",
                                 help="Share of pixels with heatmap value above 0.5.")
        metric_columns[1].metric("Attention inside the brain", f"{inside_brain:.0f}%",
                                 help="Share of total attention falling on non-background pixels. "
                                      "Low values suggest the model is relying on borders or background.")
        if inside_brain < 60:
            st.warning("Much of the attention falls outside the brain, which can mean the model "
                       "is reacting to background or framing rather than anatomy.")
        st.caption(
            "Red = regions that most increased this class's score in the EfficientNet branch. "
            "Grad-CAM does not cover the ViT branch and shows correlation, not cause."
        )

section("image-analysis", "Image analysis",
        "What the model sees and whether the input looks like its training data")
with st.container():
    height, width = original_gray.shape
    coverage = float((original_gray > 10).mean() * 100)
    foreground = original_gray[original_gray > 10]
    contrast = float(foreground.std()) if foreground.size else 0.0
    mean_intensity = float(foreground.mean()) if foreground.size else 0.0
    aspect = width / height

    metric_columns = st.columns(4)
    metric_columns[0].metric("Original size", f"{width} × {height}")
    metric_columns[1].metric("Brain coverage", f"{coverage:.0f}%",
                             help="Share of pixels brighter than the background.")
    metric_columns[2].metric("Mean intensity", f"{mean_intensity:.0f} / 255")
    metric_columns[3].metric("Contrast (std)", f"{contrast:.0f}")

    view_column, hist_column = st.columns([1, 1.4], gap="large")
    with view_column:
        st.image(model_view, caption="What the model sees (224 × 224, grayscale)",
                 use_container_width=True)
    with hist_column:
        st.markdown("##### Intensity distribution")
        st.plotly_chart(intensity_histogram(original_gray), use_container_width=True,
                        config={"displayModeBar": False})

    st.markdown("##### Input quality checks")
    checks = [
        (not is_colour, "Grayscale input",
         "The image looks like a colour image. The model was trained on grayscale MRI."),
        (coverage >= 10, "Enough brain tissue visible",
         "Very little of the image is brain tissue; the slice may be near the edge of the head."),
        (contrast >= 20, "Usable contrast",
         "Contrast is very low, which can make the prediction unreliable."),
        (0.75 <= aspect <= 1.33, "Near-square shape",
         f"Aspect ratio is {aspect:.2f}; resizing to 224 × 224 will stretch the anatomy."),
    ]
    for passed, label, problem in checks:
        if passed:
            st.markdown(f'<span class="pill" style="background:#10B981">Pass</span>&nbsp; {label}',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="pill" style="background:#F59E0B">Check</span>&nbsp; '
                        f'<b>{label}</b> — {problem}', unsafe_allow_html=True)

if is_volume:
    section("volume-insights", "Volume insights",
            "How the prediction holds up across the 3D scan")
    with st.container():
        metric_columns = st.columns(4)
        metric_columns[0].metric("Volume shape", " × ".join(str(s) for s in volume.shape))
        metric_columns[1].metric("Voxel size (mm)", " × ".join(f"{z:.2f}" for z in voxel_sizes))
        metric_columns[2].metric("Slicing axis", axis)
        metric_columns[3].metric("Selected slice", f"{slice_index} / {volume.shape[axis] - 1}")

        st.markdown("##### Orthogonal middle slices")
        ortho_columns = st.columns(3)
        for column, ortho_axis in zip(ortho_columns, [0, 1, 2]):
            try:
                ortho = normalize_mri_slice(
                    take_slice(volume, ortho_axis, volume.shape[ortho_axis] // 2)
                )
                column.image(np.rot90(np.asarray(ortho)), caption=f"Axis {ortho_axis} middle slice",
                             use_container_width=True)
            except ValueError:
                column.info(f"Axis {ortho_axis} middle slice is empty.")

        if run_sweep:
            st.markdown("##### Prediction across neighbouring slices")
            depth = volume.shape[axis]
            candidate_indices = range(
                max(0, slice_index - sweep_span * sweep_step),
                min(depth, slice_index + sweep_span * sweep_step + 1),
                sweep_step,
            )
            sweep_indices, sweep_arrays = [], []
            for index in candidate_indices:
                try:
                    sweep_arrays.append(np.asarray(normalize_mri_slice(take_slice(volume, axis, index))))
                    sweep_indices.append(index)
                except ValueError:
                    continue

            if not sweep_arrays:
                st.info("No neighbouring slices contain usable brain tissue.")
            else:
                with st.spinner(f"Running the model on {len(sweep_arrays)} slices…"):
                    sweep_probabilities = cached_predict(model, device, sweep_arrays)

                mean_probabilities = sweep_probabilities.mean(axis=0)
                consensus_class = CLASS_NAMES[int(np.argmax(mean_probabilities))]
                agreement = float((sweep_probabilities.argmax(axis=1) == predicted_index).mean() * 100)
                stability = float(100 - sweep_probabilities[:, predicted_index].std() * 100)

                metric_columns = st.columns(4)
                metric_columns[0].metric("Slices analyzed", len(sweep_indices))
                metric_columns[1].metric("Consensus class", consensus_class,
                                         f"{mean_probabilities.max() * 100:.1f}% avg", delta_color="off")
                metric_columns[2].metric(f"Slices agreeing with {predicted_class}", f"{agreement:.0f}%")
                metric_columns[3].metric("Stability", f"{stability:.0f}/100",
                                         help=f"100 − standard deviation (in points) of the "
                                              f"{predicted_class} probability across slices.")

                st.plotly_chart(slice_trend_chart(sweep_indices, sweep_probabilities, slice_index),
                                use_container_width=True, config={"displayModeBar": False})

                if consensus_class != predicted_class:
                    st.warning(
                        f"The selected slice predicts **{predicted_class}**, but the average over "
                        f"nearby slices favours **{consensus_class}**. The single-slice result "
                        "may not represent the whole scan."
                    )
                else:
                    st.success(f"Neighbouring slices agree on **{consensus_class}** on average.")

section("about-the-model", "About the model",
        "Architecture, performance and limitations")
with st.container():
    st.markdown("##### Architecture")
    arch_columns = st.columns(3)
    with arch_columns[0]:
        card("EfficientNet-B4 branch", "Convolutional network that captures local texture "
             "and shape detail. Produces 1,792 features.")
    with arch_columns[1]:
        card("ViT-B/16 branch", "Vision Transformer that splits the image into 16 × 16 patches "
             "and models global relationships between them. Produces 768 features.")
    with arch_columns[2]:
        card("Fusion classifier", "The 2,560 combined features pass through a 512-unit layer "
             "with dropout (0.4) into 3 output classes.")

    st.markdown("##### Performance")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Test accuracy", f"{TEST_ACCURACY * 100:.2f}%")
    metric_columns[1].metric("Chance level (3 classes)", "33.33%")
    if saved_val_accuracy is not None:
        metric_columns[2].metric("Validation accuracy", f"{saved_val_accuracy * 100:.2f}%")
    if saved_epoch is not None:
        metric_columns[3].metric("Best epoch", saved_epoch)

    st.markdown("##### Preprocessing")
    st.markdown(
        "- NIfTI volumes: the selected slice (middle of axis 2 by default) is normalized "
        "using the 1st–99th percentile of non-zero voxels\n"
        "- Resized to 224 × 224, grayscale repeated to 3 channels\n"
        "- ImageNet mean/std normalization"
    )

    st.markdown("##### Limitations")
    st.markdown(
        "- Accuracy is about 61% on held-out scans: roughly 2 in 5 predictions are wrong\n"
        "- Uses a single 2D slice, while Alzheimer's-related changes are 3D\n"
        "- Softmax probabilities are not calibrated and tend to be overconfident\n"
        "- Images from other scanners or protocols than the training data may perform worse"
    )

st.divider()
st.caption(
    "Research prototype only. Not clinically validated and must not be used to diagnose "
    "or guide treatment."
)
