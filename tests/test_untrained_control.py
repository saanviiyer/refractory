"""PREREG A5: the untrained control must be able to fail.

The trap this guards against: at the default threshold an untrained spiking
population never fires, so a naive random-init control returns "nothing fits"
by construction.  That looks like the reassuring answer while being an
artefact, and a control that cannot fail is not a control.
"""
from __future__ import annotations

import inspect

import numpy as np
import torch

import spike_family
import train_arm_visual as TAV
from spiking import build_model, calibrate_threshold


def test_untrained_population_is_dead_without_calibration():
    """The premise. If this ever stops holding, A5's departure is unnecessary."""
    torch.manual_seed(90_000)
    m = build_model("snn")
    rng = np.random.default_rng(1_777)
    frames, _, mask, action, _ = TAV.make_batch(4, 8, "dark", 0, rng,
                                                variant="v7")
    with torch.no_grad():
        _, h = m(frames, mask, action, return_h=True)
    assert float(h[..., m.view_slice("s")].mean()) == 0.0


def test_calibrated_untrained_population_is_alive():
    """And with the calibration the trainer applies, it fires."""
    torch.manual_seed(90_000)
    m = build_model("snn")
    rng = np.random.default_rng(1_777)
    frames, _, mask, action, _ = TAV.make_batch(4, 8, "dark", 0, rng,
                                                variant="v7")
    calibrate_threshold(m, frames, mask, action)
    with torch.no_grad():
        _, h = m(frames, mask, action, return_h=True)
    rate = float(h[..., m.view_slice("s")].mean())
    assert 0.0 < rate < 0.5, rate


def test_analyse_applies_calibration_on_the_untrained_path():
    src = inspect.getsource(spike_family.analyse)
    body = src.split("if untrained:", 1)[1].split("else:", 1)[0]
    assert "calibrate_threshold" in body, (
        "A5 requires the untrained control to receive the same pre-gradient "
        "calibration the trainer applies, or it is dead by construction")
    assert "load_state_dict" not in body


def test_untrained_run_bypasses_the_gate_on_the_record():
    src = inspect.getsource(spike_family.main)
    assert "bypassed" in src and "untrained" in src
    # ...and only when untrained; the trained path still refuses.
    trained = src.split("else:", 1)[1]
    assert "require_pass" in trained
