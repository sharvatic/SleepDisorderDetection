"""
Sleep disorder classification — interactive demo dashboard.

Run from project root:
    streamlit run dashboard.py

Requires: dataset/metadata/manifest.csv and training_output/best_model.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training import (  # noqa: E402
    DEVICE,
    SleepDisorderCNN,
    SleepTensorDataset,
    STAGE_NAMES,
)

CHECKPOINT_PATH = PROJECT_ROOT / "training_output" / "best_model.pt"
MANIFEST_PATH = PROJECT_ROOT / "dataset" / "metadata" / "manifest.csv"
RESULTS_CSV = PROJECT_ROOT / "training_output" / "results_summary.csv"
TRAINING_CURVES = PROJECT_ROOT / "training_output" / "training_curves.png"

STAGE_FILTER = [1, 2, 3, 4, 5]


@st.cache_resource
def load_model(checkpoint_path: Path):
    try:
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    n_classes = int(ckpt["n_classes"])
    class_names = list(ckpt["class_names"])
    model = SleepDisorderCNN(n_classes=n_classes).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    meta = {
        "epoch": int(ckpt.get("epoch", 0)),
        "val_loss": float(ckpt.get("val_loss", 0.0)),
        "val_acc": float(ckpt.get("val_acc", 0.0)),
        "n_classes": n_classes,
        "class_names": class_names,
    }
    return model, meta


@st.cache_resource
def load_dataset(manifest_path: Path):
    if not manifest_path.is_file():
        return None
    return SleepTensorDataset(
        manifest_path=str(manifest_path),
        label_dir=None,
        stage_filter=STAGE_FILTER,
        disorder_filter=None,
    )


def tensor_slice_to_rgb(frame_hwc: np.ndarray) -> np.ndarray:
    """frame (H, W, 3) float [0,1] -> uint8 RGB for display."""
    x = np.clip(frame_hwc * 255.0, 0, 255).astype(np.uint8)
    return x


def predict_proba(model: torch.nn.Module, tensor_cthw: torch.Tensor) -> tuple[int, np.ndarray]:
    """tensor (1, 3, 30, 32, 32) -> predicted class index, probabilities."""
    with torch.no_grad():
        logits = model(tensor_cthw.to(DEVICE))
        probs = F.softmax(logits, dim=1).cpu().numpy().flatten()
    return int(probs.argmax()), probs


def main():
    st.set_page_config(
        page_title="Sleep Disorder Detection — Demo",
        page_icon="🌙",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .main-header { font-size: 1.85rem; font-weight: 650; letter-spacing: -0.02em;
            margin-bottom: 0.25rem; color: #e8eef7; }
        .subtle { color: #9aa7b8; font-size: 0.95rem; margin-bottom: 1.25rem; }
        div[data-testid="stMetric"] { background: linear-gradient(145deg, #1a2332 0%, #141b26 100%);
            border: 1px solid #2a3544; border-radius: 10px; padding: 0.65rem 0.85rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<p class="main-header">Sleep disorder classification — 3D CNN demo</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtle">Spatiotemporal EEG tensors (delta / alpha / beta as RGB) → disorder prediction. '
        "Trained on CAP sleep epochs (non–wake stages).</p>",
        unsafe_allow_html=True,
    )

    if not CHECKPOINT_PATH.is_file():
        st.error(f"Checkpoint not found: `{CHECKPOINT_PATH}`. Train the model or adjust the path.")
        st.stop()

    model, ckpt_meta = load_model(CHECKPOINT_PATH)
    dataset = load_dataset(MANIFEST_PATH)

    with st.sidebar:
        st.header("Model")
        st.caption(f"Checkpoint: `{CHECKPOINT_PATH.name}`")
        st.metric("Classes", ckpt_meta["n_classes"])
        st.metric("Best val accuracy", f"{ckpt_meta['val_acc']*100:.1f}%")
        st.metric("Saved at epoch", ckpt_meta["epoch"])
        st.divider()
        st.markdown("**Output labels**")
        for i, name in enumerate(ckpt_meta["class_names"]):
            st.caption(f"{i}: {name}")

    tab_overview, tab_explore = st.tabs(["Overview", "Explore sample"])

    with tab_overview:
        c1, c2, c3, c4 = st.columns(4)
        if RESULTS_CSV.is_file():
            summ = pd.read_csv(RESULTS_CSV).iloc[0]
            c1.metric("Validation accuracy", f"{float(summ['val_accuracy'])*100:.1f}%")
            c2.metric("Test accuracy", f"{float(summ['test_accuracy'])*100:.1f}%")
            c3.metric("Train epochs", f"{int(summ['train_epochs'])}")
            c4.metric("Best epoch", f"{int(summ['best_epoch'])}")
        else:
            c1.metric("Val accuracy (ckpt)", f"{ckpt_meta['val_acc']*100:.1f}%")

        st.subheader("Pipeline")
        st.markdown(
            """
1. **Input** — 30 s of bipolar EEG per epoch, band-passed into delta, alpha, and beta; STFT power per second per channel.  
2. **Spatial map** — RBF interpolation to a 32×32 scalp grid; three bands encoded as **B, G, R** channels → tensor shape **(30, 32, 32, 3)**.  
3. **Model** — 3D CNN (`SleepDisorderCNN`): conv blocks over time and space, then classifier over disorder classes.  
4. **Output** — probability distribution over disorder labels (patient-level diagnosis reflected in training labels).
            """
        )

        st.subheader("Architecture (summary)")
        st.code(
            "Input: (batch, 3, 30, 32, 32)  — channels = beta, alpha, delta\n"
            "Blocks: Conv3D + BN + ReLU + pool → … → AdaptiveAvgPool3d → MLP → logits",
            language="text",
        )

        if TRAINING_CURVES.is_file():
            st.subheader("Training curves")
            st.image(str(TRAINING_CURVES), use_container_width=True)

    with tab_explore:
        if dataset is None:
            st.warning(
                f"Dataset manifest not found at `{MANIFEST_PATH}`. "
                "Run `build_dataset.py` so `manifest.csv` exists, then refresh this page."
            )
            st.stop()

        st.subheader("Run inference on a saved epoch")
        n = len(dataset)
        manifest_view = dataset.manifest.reset_index(drop=True)

        col_a, col_b = st.columns([1, 2])
        with col_a:
            mode = st.radio("Choose sample", ["Pick index", "Random"], horizontal=True)
            if mode == "Pick index":
                idx = st.number_input("Dataset index", min_value=0, max_value=n - 1, value=0, step=1)
            else:
                if "rand_idx" not in st.session_state:
                    st.session_state["rand_idx"] = int(np.random.randint(0, n))
                if st.button("Draw random sample", type="primary"):
                    st.session_state["rand_idx"] = int(np.random.randint(0, n))
                idx = int(st.session_state["rand_idx"])

        tensor, disorder_idx, stage_idx = dataset[idx]
        row = manifest_view.iloc[idx]
        tensor_b = tensor.unsqueeze(0)
        pred_idx, probs = predict_proba(model, tensor_b)

        true_name = dataset.class_names[disorder_idx]
        pred_name = ckpt_meta["class_names"][pred_idx]
        stage_name = STAGE_NAMES.get(stage_idx, str(stage_idx))

        with col_b:
            st.markdown(
                f"**Patient:** `{row.get('patient_id', '—')}` · **Epoch:** {row.get('epoch_idx', '—')} · "
                f"**Stage:** {stage_name}"
            )

        m1, m2, m3 = st.columns(3)
        m1.metric("Ground truth", true_name)
        m2.metric("Prediction", pred_name)
        m3.metric("Match", "Yes" if pred_idx == disorder_idx else "No")

        st.subheader("Class probabilities")
        prob_df = pd.DataFrame({"Class": ckpt_meta["class_names"], "Probability": probs})
        st.bar_chart(prob_df.set_index("Class"))

        st.subheader("Spatiotemporal tensor (RGB = delta, alpha, beta)")
        np_t = tensor.numpy().transpose(1, 2, 3, 0)
        slots = [0, 5, 10, 15, 20, 29]
        cols = st.columns(len(slots))
        for c, t_i in zip(cols, slots):
            with c:
                st.caption(f"t = {t_i}s")
                st.image(tensor_slice_to_rgb(np_t[t_i]), use_container_width=True)

        st.caption(
            "Each frame is a scalp map for one second: red ≈ beta, green ≈ alpha, blue ≈ delta "
            "(after global normalization from dataset build)."
        )


if __name__ == "__main__":
    main()
