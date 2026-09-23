"""The gate refuses, and the refusal is wired into the analyses.

A gate that warns is a gate that gets scrolled past.  The failure it exists to
catch --- a prior that was never trained, a surrogate that never carried
gradient --- looks healthy in the loss curve and in every statistic of the term
itself, so the only useful behaviour is to stop.
"""
from __future__ import annotations

import json

import pytest
import torch

import blackout_validity as bv
import gate_blackout as gb
from spiking import SpikingArmWM


def write(tmp_path, name, payload):
    p = tmp_path / name
    p.write_text(json.dumps(payload))
    return p


def test_missing_report_refuses():
    with pytest.raises(SystemExit):
        gb.require_pass("runs/does_not_exist.json")


@pytest.mark.parametrize("verdict", ["FAIL", "PARTIAL"])
def test_failing_report_refuses(tmp_path, verdict):
    p = write(tmp_path, "gate.json", {"verdict": verdict, "n_models": 8,
                                      "n_beating_persistence": 0,
                                      "median_ratio": 2.1})
    with pytest.raises(SystemExit):
        gb.require_pass(p)


def test_passing_report_is_returned(tmp_path):
    payload = {"verdict": "PASS", "n_models": 8, "n_beating_persistence": 8,
               "median_ratio": 0.68}
    assert gb.require_pass(write(tmp_path, "gate.json", payload)) == payload


def test_analyses_call_the_gate_before_reading_anything():
    import inspect

    import spike_family
    import spike_torus
    for module in (spike_torus, spike_family):
        source = inspect.getsource(module.main)
        assert "require_pass" in source, module.__name__
        body = source.split("\n")
        first = next(i for i, line in enumerate(body)
                     if "require_pass" in line)
        later = [line for line in body[first + 1:] if "analyse(" in line]
        assert later, "the gate must be checked before the analysis, not after"


def test_persistence_baseline_holds_the_last_visible_frame():
    frames = torch.arange(2 * 4 * 1 * 1 * 1, dtype=torch.float32) \
        .reshape(2, 4, 1, 1, 1)
    mask = torch.tensor([[0.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 1.0]])
    held, valid = bv._last_visible(frames, mask)
    # Row 0 sees frame 0, then holds it through the blackout.
    assert held[0, 1, 0, 0, 0] == frames[0, 0, 0, 0, 0]
    assert held[0, 2, 0, 0, 0] == frames[0, 0, 0, 0, 0]
    assert held[0, 3, 0, 0, 0] == frames[0, 3, 0, 0, 0]
    # Row 1 has seen nothing until t=2, so earlier steps are not scored.
    assert not valid[1, 0] and not valid[1, 1] and valid[1, 2]


def test_gate_source_accepts_the_spiking_kinds():
    src = gb.SpikingArmSource("v6")
    assert isinstance(src.model("snn"), SpikingArmWM)
    assert src.model("gru").__class__.__name__ == "VisualGRU"


def test_calibration_expectations_match_the_parents_published_verdicts():
    # If these ever change, the calibration is no longer a check on the gate.
    assert gb.CALIBRATION[("arm_v6", "gru")] == "PASS"
    assert gb.CALIBRATION[("rssm_v6", "rssm")] == "FAIL"
