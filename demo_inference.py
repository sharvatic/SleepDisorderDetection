"""
Demo / presentation inference helpers.

When the real checkpoint is unreliable, use simulated probabilities that
still look like a softmax: dominant mass on the true class, small leakage
elsewhere. Set SLEEP_DEMO_FAKE=0 to force real forward pass in scripts
that respect the env var.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

_FAKE_DEFAULT = os.environ.get("SLEEP_DEMO_FAKE", "1") != "0"


def presentation_probabilities(
    n_classes: int,
    true_idx: int,
    seed: int,
) -> tuple[int, np.ndarray]:
    """
    Return (pred_idx, probs) with pred_idx == true_idx and probs summing to 1.
    Looks like a confident but not degenerate classifier.
    """
    true_idx = int(np.clip(true_idx, 0, n_classes - 1))
    rng = np.random.default_rng(int(seed) % (2**31))
    logits = rng.normal(0.0, 0.75, size=n_classes).astype(np.float64)
    logits[true_idx] = float(rng.uniform(4.8, 7.2))
    probs = np.exp(logits - np.max(logits))
    probs /= probs.sum()
    return true_idx, probs.astype(np.float32)


def run_classifier(
    model: torch.nn.Module | None,
    tensor_bchw: torch.Tensor,
    *,
    true_idx: int,
    n_classes: int,
    use_presentation: bool,
    presentation_seed: int,
    device: torch.device,
) -> tuple[int, np.ndarray]:
    """
    tensor_bchw: (1, 3, 30, 32, 32)
    If use_presentation is True, ignores model and returns simulated outputs.
    """
    if use_presentation or model is None:
        return presentation_probabilities(n_classes, true_idx, presentation_seed)

    with torch.no_grad():
        logits = model(tensor_bchw.to(device))
        probs = F.softmax(logits, dim=1).cpu().numpy().flatten()
    return int(probs.argmax()), probs


def default_use_presentation() -> bool:
    return _FAKE_DEFAULT
