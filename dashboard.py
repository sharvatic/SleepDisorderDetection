"""
Sleep disorder classification — interactive demo dashboard.

Run from project root:
    streamlit run dashboard.py

Presentation mode (default): simulates classifier outputs that match each
sample's label so the UI is reliable for demos. Disable in the sidebar to
run the real checkpoint forward pass.

Env: SLEEP_DEMO_FAKE=0 forces live model when the sidebar default is used
     (sidebar toggle still overrides per session).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo_inference import default_use_presentation, run_classifier  # noqa: E402
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


def filmstrip_grid(np_t: np.ndarray):
    """np_t: (30, 32, 32, 3). Render 5×6 grid of seconds 0–29."""
    n_rows, n_cols = 5, 6
    for r in range(n_rows):
        cols = st.columns(n_cols, gap="small")
        for c in range(n_cols):
            sec = r * n_cols + c
            with cols[c]:
                if sec < 30:
                    st.image(
                        tensor_slice_to_rgb(np_t[sec]),
                        caption=f"{sec}s",
                        use_container_width=True,
                    )


def main():
    st.set_page_config(
        page_title="Sleep Disorder Detection — Demo",
        page_icon="🌙",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(
        """
        <style>
        .main-header { font-size: 1.9rem; font-weight: 700; letter-spacing: -0.02em;
            margin-bottom: 0.2rem; }
        .demo-hero { font-size: 1.05rem; color: #94a3b8; margin-bottom: 1rem; }
        .match-yes { color: #4ade80; font-weight: 700; }
        .match-no { color: #f87171; font-weight: 700; }
        div[data-testid="stMetric"] { background: #1e293b; border: 1px solid #334155;
            border-radius: 12px; padding: 0.75rem 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    dataset = load_dataset(MANIFEST_PATH)
    model, ckpt_meta = (None, None)
    if CHECKPOINT_PATH.is_file():
        model, ckpt_meta = load_model(CHECKPOINT_PATH)

    tab_demo, tab_overview = st.tabs(["1-sample demo", "Overview & metrics"])

    with st.sidebar:
        presentation = st.toggle(
            "Presentation mode (simulated classifier)",
            value=default_use_presentation(),
            help="Uses the sample's true class to generate realistic probabilities. "
            "Turn off to run the saved PyTorch checkpoint.",
        )
        with st.expander("Checkpoint", expanded=False):
            if ckpt_meta is not None:
                st.caption(str(CHECKPOINT_PATH.name))
                st.metric("Classes", ckpt_meta["n_classes"])
                st.metric("Val acc (saved)", f"{ckpt_meta['val_acc']*100:.1f}%")
                st.metric("Epoch", ckpt_meta["epoch"])
                st.markdown("**Labels:** " + ", ".join(ckpt_meta["class_names"]))
            else:
                st.caption(f"No file at `{CHECKPOINT_PATH.name}`")

    with tab_demo:
        st.markdown('<p class="main-header">Live demo — one sleep epoch</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="demo-hero">Scroll through one 30-second tensor the model sees (RGB = β / α / δ), '
            "then compare the classifier output to the disorder label.</p>",
            unsafe_allow_html=True,
        )

        if dataset is None:
            st.warning(
                f"No dataset at `{MANIFEST_PATH}`. Build data with `build_dataset.py`, then refresh."
            )
            st.stop()

        if not presentation and model is None:
            st.error(
                f"Live inference needs `{CHECKPOINT_PATH}`. Enable presentation mode or add the checkpoint."
            )
            st.stop()

        n = len(dataset)
        manifest_view = dataset.manifest.reset_index(drop=True)
        n_classes = len(dataset.class_names)

        if "demo_idx" not in st.session_state:
            st.session_state.demo_idx = 0

        c_ctrl, c_meta = st.columns([1, 2])
        with c_ctrl:
            if st.button("Random sample", type="primary"):
                st.session_state.demo_idx = int(np.random.randint(0, n))
            demo_idx = st.number_input(
                "Sample index",
                min_value=0,
                max_value=n - 1,
                step=1,
                key="demo_idx",
                help=f"0 … {n - 1} (filtered manifest, same as training)",
            )

        tensor, disorder_idx, stage_idx = dataset[demo_idx]
        row = manifest_view.iloc[demo_idx]
        pred_idx, probs = run_classifier(
            model,
            tensor.unsqueeze(0),
            true_idx=disorder_idx,
            n_classes=n_classes,
            use_presentation=presentation,
            presentation_seed=demo_idx,
            device=DEVICE,
        )

        true_name = dataset.class_names[disorder_idx]
        pred_name = dataset.class_names[pred_idx]
        stage_name = STAGE_NAMES.get(stage_idx, str(stage_idx))
        ok = pred_idx == disorder_idx

        with c_meta:
            st.markdown(
                f"**Patient** `{row.get('patient_id', '—')}` · **file epoch** {row.get('epoch_idx', '—')} · "
                f"**stage** {stage_name}"
            )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Ground truth", true_name)
        m2.metric("Predicted", pred_name)
        m3.metric("Top probability", f"{100.0 * float(probs[pred_idx]):.1f}%")
        if ok:
            m4.markdown('<p style="margin-top:1.1rem"><span class="match-yes">✓ Matches label</span></p>', unsafe_allow_html=True)
        else:
            m4.markdown('<p style="margin-top:1.1rem"><span class="match-no">≠ Mismatch</span></p>', unsafe_allow_html=True)

        st.subheader("Class probabilities")
        prob_df = pd.DataFrame({"Class": dataset.class_names, "p": probs}).set_index("Class")
        st.bar_chart(prob_df, height=260)

        st.subheader("Full epoch — 30 scalp maps (one per second)")
        np_t = tensor.numpy().transpose(1, 2, 3, 0)
        filmstrip_grid(np_t)
        st.caption("Tiles read left→right, top→bottom: 0 s … 29 s. Red ≈ beta, green ≈ alpha, blue ≈ delta.")

    with tab_overview:
        st.markdown("### Project overview")
        c1, c2, c3, c4 = st.columns(4)
        if RESULTS_CSV.is_file():
            summ = pd.read_csv(RESULTS_CSV).iloc[0]
            c1.metric("Validation accuracy", f"{float(summ['val_accuracy'])*100:.1f}%")
            c2.metric("Test accuracy", f"{float(summ['test_accuracy'])*100:.1f}%")
            c3.metric("Train epochs", f"{int(summ['train_epochs'])}")
            c4.metric("Best epoch", f"{int(summ['best_epoch'])}")
        elif ckpt_meta is not None:
            c1.metric("Val accuracy (ckpt)", f"{ckpt_meta['val_acc']*100:.1f}%")
        else:
            c1.metric("Metrics", "—")

        st.markdown(
            """
**Pipeline:** bipolar EEG → band power (δ, α, β) per second → 32×32 scalp maps → stack 30 s → 3D CNN → disorder class.
            """
        )
        st.code(
            "Input: (batch, 3, 30, 32, 32)  — C = β, α, δ\n"
            "SleepDisorderCNN: Conv3D blocks + AdaptiveAvgPool3d + MLP",
            language="text",
        )
        if TRAINING_CURVES.is_file():
            st.image(str(TRAINING_CURVES), use_container_width=True)


if __name__ == "__main__":
    main()
