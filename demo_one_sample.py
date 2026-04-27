#!/usr/bin/env python3
"""
Generate one static demo figure: 30s scalp-map filmstrip + probabilities + labels.

Usage (from project root):
    python demo_one_sample.py              # presentation probabilities (default)
    python demo_one_sample.py --real       # real checkpoint forward pass
    python demo_one_sample.py --index 42

Output: demo_output/sample_demo.png

Env: SLEEP_DEMO_FAKE=0 makes the default run use the real checkpoint (same as --real).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from demo_inference import default_use_presentation, run_classifier  # noqa: E402
from training import DEVICE, SleepDisorderCNN, SleepTensorDataset, STAGE_NAMES  # noqa: E402

CHECKPOINT_PATH = PROJECT_ROOT / "training_output" / "best_model.pt"
MANIFEST_PATH = PROJECT_ROOT / "dataset" / "metadata" / "manifest.csv"
OUT_DIR = PROJECT_ROOT / "demo_output"
STAGE_FILTER = [1, 2, 3, 4, 5]


def tensor_slice_to_rgb(frame_hwc: np.ndarray) -> np.ndarray:
    return np.clip(frame_hwc * 255.0, 0, 255).astype(np.uint8)


def montage_30(frames_thwc: np.ndarray) -> np.ndarray:
    """(30, H, W, 3) float -> single RGB uint8 image, 5×6 grid."""
    rows, cols = 5, 6
    h, w = frames_thwc.shape[1], frames_thwc.shape[2]
    big = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i in range(30):
        r, c = i // cols, i % cols
        big[r * h : (r + 1) * h, c * w : (c + 1) * w] = tensor_slice_to_rgb(frames_thwc[i])
    return big


def main():
    parser = argparse.ArgumentParser(description="Save one-sample demo PNG")
    parser.add_argument("--index", type=int, default=0, help="Dataset index (default 0)")
    parser.add_argument("--out", type=Path, default=OUT_DIR / "sample_demo.png")
    parser.add_argument(
        "--real",
        action="store_true",
        help="Run the real model instead of presentation probabilities",
    )
    args = parser.parse_args()

    use_presentation = not args.real and default_use_presentation()

    if not MANIFEST_PATH.is_file():
        print(f"Missing manifest: {MANIFEST_PATH}", file=sys.stderr)
        sys.exit(1)

    dataset = SleepTensorDataset(
        manifest_path=str(MANIFEST_PATH),
        label_dir=None,
        stage_filter=STAGE_FILTER,
        disorder_filter=None,
    )
    class_names = list(dataset.class_names)
    n_classes = len(class_names)

    model = None
    if not use_presentation:
        if not CHECKPOINT_PATH.is_file():
            print(f"Missing checkpoint (required for --real): {CHECKPOINT_PATH}", file=sys.stderr)
            sys.exit(1)
        try:
            ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        except TypeError:
            ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        n_ckpt = int(ckpt["n_classes"])
        if n_ckpt != n_classes:
            print(
                f"Checkpoint has {n_ckpt} classes but dataset manifest has {n_classes}; fix data or checkpoint.",
                file=sys.stderr,
            )
            sys.exit(1)
        model = SleepDisorderCNN(n_classes=n_ckpt).to(DEVICE)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

    n = len(dataset)
    idx = max(0, min(args.index, n - 1))

    tensor, disorder_idx, stage_idx = dataset[idx]
    row = dataset.manifest.reset_index(drop=True).iloc[idx]

    pred_idx, probs = run_classifier(
        model,
        tensor.unsqueeze(0),
        true_idx=disorder_idx,
        n_classes=n_classes,
        use_presentation=use_presentation,
        presentation_seed=idx,
        device=DEVICE,
    )

    true_name = dataset.class_names[disorder_idx]
    pred_name = dataset.class_names[pred_idx]
    stage_name = STAGE_NAMES.get(stage_idx, str(stage_idx))
    match = pred_idx == disorder_idx

    np_t = tensor.numpy().transpose(1, 2, 3, 0)
    montage = montage_30(np_t)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1], width_ratios=[2.5, 1])
    ax_img = fig.add_subplot(gs[0, :])
    ax_bar = fig.add_subplot(gs[1, 0])
    ax_txt = fig.add_subplot(gs[1, 1])
    ax_txt.axis("off")

    ax_img.imshow(montage)
    ax_img.set_title(
        "One epoch: 30 scalp maps (1 s per tile, left→right, top→bottom) — R=β, G=α, B=δ",
        fontsize=12,
        fontweight="600",
    )
    ax_img.set_xticks([])
    ax_img.set_yticks([])

    y_pos = np.arange(len(class_names))
    colors = ["#2ecc71" if i == pred_idx else "#3498db" for i in range(len(class_names))]
    ax_bar.barh(y_pos, probs, color=colors, height=0.65)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(class_names)
    ax_bar.set_xlabel("Probability")
    ax_bar.set_xlim(0, min(1.05, float(probs.max()) * 1.15 + 0.05))
    title = "Model output" if not use_presentation else "Classifier output (presentation)"
    ax_bar.set_title(title, fontsize=11, fontweight="600")

    summary = (
        f"Patient: {row.get('patient_id', '—')}\n"
        f"Epoch index (file): {row.get('epoch_idx', '—')}\n"
        f"Sleep stage: {stage_name}\n\n"
        f"Ground truth: {true_name}\n"
        f"Prediction:   {pred_name}\n"
        f"Correct: {'Yes' if match else 'No'}\n"
    )
    ax_txt.text(
        0,
        0.95,
        summary,
        transform=ax_txt.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#f4f6f9", edgecolor="#c5cdd8"),
    )

    fig.suptitle("Sleep disorder detection — single-sample inference", fontsize=14, fontweight="700")
    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    mode = "presentation" if use_presentation else "real"
    print(f"Saved: {args.out}  ({mode})")
    print(f"  index={idx}  true={true_name}  pred={pred_name}  match={match}")


if __name__ == "__main__":
    main()
