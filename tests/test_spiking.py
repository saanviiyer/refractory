"""Structural invariants of the spiking core and its rate twin."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from spiking import (PRIMARY_VIEW, VIEW_ORDER, RateArmWM, SpikingArmWM,
                     build_model, calibrate_threshold, n_params, soft_spike,
                     spike)
from visual_models import VisualGRU


def batch(B=3, T=4):
    torch.manual_seed(0)
    return (torch.rand(B, T, 3, 16, 32), torch.zeros(B, T),
            torch.rand(B, T, 2) * 2 - 1)


def test_state_views_partition_the_state():
    m = SpikingArmWM(neurons=8)
    covered = np.zeros(m.hidden, dtype=int)
    for name in VIEW_ORDER:
        sl = m.view_slice(name)
        covered[sl] += 1
    assert covered.min() == 1 and covered.max() == 1
    assert m.hidden == len(VIEW_ORDER) * m.neurons


def test_spikes_are_binary_and_the_trace_is_not():
    m = SpikingArmWM(neurons=8)
    frames, mask, action = batch()
    calibrate_threshold(m, frames, mask, action)
    _, hidden = m(frames, mask, action, return_h=True)
    s = hidden[..., m.view_slice("s")]
    assert torch.isin(s, torch.tensor([0.0, 1.0])).all()
    assert s.sum() > 0, "a model that never fires cannot carry a coordinate"
    r = hidden[..., m.view_slice("r")]
    assert not torch.isin(r, torch.tensor([0.0, 1.0])).all()


def test_calibration_wakes_a_population_that_starts_silent():
    """The default threshold against this encoder's scale fires nothing.

    That is the spiking form of the failure this line of work has already paid
    for: a dead path that trains, converges and reports a number.  Calibration
    is not a nicety, so the test asserts the silence first.
    """
    m = SpikingArmWM(neurons=32)
    frames, mask, action = batch(B=4, T=8)
    with torch.no_grad():
        _, before = m(frames, mask, action, return_h=True)
    assert before[..., m.view_slice("s")].mean() == 0.0
    report = calibrate_threshold(m, frames, mask, action, quantile=0.90)
    with torch.no_grad():
        _, after = m(frames, mask, action, return_h=True)
    rate = float(after[..., m.view_slice("s")].mean())
    assert 0.0 < rate < 0.5, rate
    assert report["n_degenerate_neurons"] == 0


def test_rate_twin_emits_continuously_and_costs_the_same():
    snn, rate = SpikingArmWM(neurons=16), RateArmWM(neurons=16)
    assert n_params(snn) == n_params(rate)
    frames, mask, action = batch()
    _, h = rate(frames, mask, action, return_h=True)
    s = h[..., rate.view_slice("s")]
    assert ((s > 0) & (s < 1)).any()


def test_surrogate_forward_is_the_twin_forward():
    x = torch.linspace(-2, 2, 41)
    hard, soft = spike(x, 4.0), soft_spike(x, 4.0)
    assert torch.equal(hard, (x > 0).float())
    assert torch.all((soft > 0) & (soft < 1))
    # The twin is the surrogate's own forward, so the two agree in the limit
    # everywhere except at the threshold itself, where the step is undefined
    # and the sigmoid is one half.
    away = x[x.abs() > 1e-6]
    assert torch.allclose(soft_spike(away, 1e4), (away > 0).float(), atol=1e-6)
    assert soft_spike(torch.zeros(1), 1e4).item() == 0.5


def test_surrogate_gradient_is_finite_and_peaks_at_the_threshold():
    x = torch.linspace(-2, 2, 41, requires_grad=True)
    spike(x, 4.0).sum().backward()
    g = x.grad
    assert torch.isfinite(g).all()
    assert int(g.argmax()) == 20 and g[20] > g[0]


def test_reset_clears_the_potential_after_a_spike():
    """The carried potential is exactly the term the spike removes.

    Compared against an otherwise identical state that did not spike, holding
    everything else fixed: the recurrent weights are zeroed so the spike cannot
    act through the drive, leaving the reset as the only channel.
    """
    m = SpikingArmWM(neurons=4)
    with torch.no_grad():
        m.recurrent.weight.zero_()
        m.logit_beta.fill_(2.0)
    beta = float(torch.sigmoid(m.logit_beta[0].detach()))
    z, mask, action = torch.zeros(2, 96), torch.zeros(2), torch.zeros(2, 2)
    h = torch.randn(2, m.hidden)
    quiet, fired = h.clone(), h.clone()
    quiet[..., m.view_slice("s")] = 0.0
    fired[..., m.view_slice("s")] = 1.0
    with torch.no_grad():
        v_quiet = m.advance(z, mask, action, quiet)[..., m.view_slice("v")]
        v_fired = m.advance(z, mask, action, fired)[..., m.view_slice("v")]
    carried = beta * h[..., m.view_slice("v")]
    assert torch.allclose(v_quiet - v_fired, carried, atol=1e-5)


def test_step_and_forward_agree():
    m = SpikingArmWM(neurons=8)
    m.eval()
    frames, mask, action = batch()
    with torch.no_grad():
        _, hidden = m(frames, mask, action, return_h=True)
        h = m.initial_state(frames.shape[0], frames)
        for t in range(frames.shape[1]):
            h = m.step(frames[:, t], mask[:, t], action[:, t], h)
    assert torch.allclose(h, hidden[:, -1], atol=1e-5)


def test_readout_reads_the_primary_view_only():
    m = SpikingArmWM(neurons=8)
    m.eval()
    h = torch.randn(2, m.hidden)
    other = h.clone()
    for name in VIEW_ORDER:
        if name != PRIMARY_VIEW:
            other[..., m.view_slice(name)] = torch.randn(2, m.neurons)
    with torch.no_grad():
        assert torch.allclose(m.readout(h), m.readout(other))


@pytest.mark.parametrize("kind,expect_within", [("snn", 0.15),
                                                ("snn_wide", 0.02)])
def test_budgets_are_matched_to_the_gru(kind, expect_within):
    gru = n_params(VisualGRU())
    got = n_params(build_model(kind))
    assert abs(got - gru) / gru < expect_within, (kind, got, gru)


def test_width_matched_view_equals_the_gru_hidden_width():
    assert SpikingArmWM(neurons=128).neurons == VisualGRU().hidden


def test_margins_are_raw_and_unsummarised():
    m = SpikingArmWM(neurons=8)
    frames, mask, action = batch(B=2, T=5)
    out = m.margins(frames, mask, action)
    assert out.shape == (2, 5, 8)
    assert torch.isfinite(out).all()


# --- PREREG A1: the graded control ------------------------------------------

def test_graded_control_is_budget_matched_and_not_discrete():
    from spiking import GradedRateArmWM
    soft, rate, snn = (GradedRateArmWM(neurons=16), RateArmWM(neurons=16),
                       SpikingArmWM(neurons=16))
    assert n_params(soft) == n_params(rate) == n_params(snn)
    assert soft.discrete is False
    assert soft.emission_width < soft.surrogate_width


def test_graded_control_stays_graded_where_the_twin_saturates():
    """The whole point of A1, asserted on the margins the pilot measured.

    The v6 twin's median |margin| was 2.78, where sigmoid(4x) is 0.99998. The
    control has to still be emitting an intermediate value there, or it is the
    same partial control under a new name.
    """
    from spiking import GradedRateArmWM
    soft = GradedRateArmWM(neurons=4)
    margins = torch.tensor([-2.78, -1.47, 0.0, 1.47, 2.78])
    emitted = soft._emit(margins)
    assert ((emitted > 0.02) & (emitted < 0.98)).all(), emitted
    saturating = RateArmWM(neurons=4)._emit(margins)
    assert ((saturating < 0.02) | (saturating > 0.98)).sum() >= 4


def test_graded_control_backward_is_its_own_forward_derivative():
    """No surrogate: unlike the spiking model, forward and backward agree."""
    from spiking import GradedRateArmWM
    soft = GradedRateArmWM(neurons=4)
    x = torch.linspace(-3, 3, 25, requires_grad=True)
    soft._emit(x).sum().backward()
    w = soft.emission_width
    expected = w * torch.sigmoid(w * x) * (1 - torch.sigmoid(w * x))
    assert torch.allclose(x.grad, expected.detach(), atol=1e-6)
    # And it is a far stronger gradient than the surrogate at those margins.
    assert float(x.grad.min()) > 1.0 / (1.0 + 4.0 * 3.0) ** 2
