"""The view-restricted transforms, and their agreement with the parent's.

The parent rotates the whole hidden vector because a GRU has one.  This
project's transform rotates a named block and leaves the rest of the state
alone.  The parent's transform must therefore be recoverable exactly as the
``view="all"`` case, on a model whose state is a single block --- otherwise the
comparison the whole project rests on is between two different estimators.
"""
from __future__ import annotations

import numpy as np
import torch

import arm_fast
import spike_fast as sf
from spiking import VIEW_ORDER, RateArmWM, SpikingArmWM


def plane(dim, seed=0):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(dim, 2)))
    return q[:, :2]


class _SingleBlock(SpikingArmWM):
    """A model whose view layout is one block, so ``all`` is the whole state."""

    def views(self):
        return {"v": slice(0, self.hidden)}


def test_view_all_matches_the_parents_whole_state_rotation():
    m = _SingleBlock(neurons=6)
    h = torch.randn(5, m.hidden, dtype=torch.float64)
    mu = h.mean(0, keepdim=True)
    P = torch.as_tensor(plane(m.hidden), dtype=torch.float64)
    for phi in (0.0, 0.37, -1.2):
        want = arm_fast.rotate_states(h, mu, P, phi)
        got = sf.rotate_view(m, h, mu, P.numpy(), phi, "all")
        assert torch.allclose(want, got, atol=1e-12)


def test_view_all_matches_the_parent_for_per_state_angles():
    m = _SingleBlock(neurons=6)
    h = torch.randn(7, m.hidden, dtype=torch.float64)
    mu = h.mean(0, keepdim=True)
    P = torch.as_tensor(plane(m.hidden, 1), dtype=torch.float64)
    phi = torch.linspace(-0.5, 0.5, 7, dtype=torch.float64)
    want = arm_fast.rotate_states(h, mu, P, phi)
    got = sf.rotate_view(m, h, mu, P.numpy(), phi, "all")
    assert torch.allclose(want, got, atol=1e-12)


def test_translation_view_all_matches_the_parent():
    m = _SingleBlock(neurons=6)
    h = torch.randn(4, m.hidden, dtype=torch.float64)
    d = np.linalg.qr(np.random.default_rng(2).normal(size=(m.hidden, 1)))[0][:, 0]
    for shift in (0.0, 0.9):
        want = arm_fast.translate_states(h, d, shift)
        got = sf.translate_view(m, h, d, shift, "all")
        assert torch.allclose(want, got, atol=1e-12)


def test_rotation_touches_only_its_own_view():
    m = SpikingArmWM(neurons=6)
    h = torch.randn(5, m.hidden, dtype=torch.float64)
    mu = h.mean(0, keepdim=True)
    P = plane(m.neurons, 3)
    for view in VIEW_ORDER:
        out = sf.rotate_view(m, h, mu, P, 0.8, view)
        for other in VIEW_ORDER:
            block_in = h[..., m.view_slice(other)]
            block_out = out[..., m.view_slice(other)]
            if other == view:
                assert not torch.allclose(block_in, block_out)
            else:
                assert torch.equal(block_in, block_out), (view, other)


def test_zero_rotation_is_the_identity():
    m = SpikingArmWM(neurons=6)
    h = torch.randn(5, m.hidden, dtype=torch.float64)
    mu = h.mean(0, keepdim=True)
    for view in VIEW_ORDER + ("all",):
        dim = m.hidden if view == "all" else m.neurons
        out = sf.rotate_view(m, h, mu, plane(dim, 4), 0.0, view)
        assert torch.allclose(h, out, atol=1e-12)


def test_rotations_compose_within_a_view():
    m = SpikingArmWM(neurons=6)
    h = torch.randn(5, m.hidden, dtype=torch.float64)
    mu = h.mean(0, keepdim=True)
    P = plane(m.neurons, 5)
    once = sf.rotate_view(m, h, mu, P, 0.3, "i")
    twice = sf.rotate_view(m, once, mu, P, 0.4, "i")
    direct = sf.rotate_view(m, h, mu, P, 0.7, "i")
    assert torch.allclose(twice, direct, atol=1e-10)


def test_coherent_rederives_spikes_only_where_it_is_defined():
    m = SpikingArmWM(neurons=6)
    m.eval()
    h = torch.randn(5, m.hidden)
    with torch.no_grad():
        h[..., m.view_slice("s")] = m._emit(
            h[..., m.view_slice("v")] - m.threshold)
    mu = h.mean(0, keepdim=True)
    P = plane(m.neurons, 6)
    coh = sf.rotate_view(m, h, mu, P, 1.1, "v", coherent=True)
    plain = sf.rotate_view(m, h, mu, P, 1.1, "v", coherent=False)
    assert not torch.equal(coh[..., m.view_slice("s")],
                           plain[..., m.view_slice("s")])
    assert sf.coherent_is_effective("v", True)
    # For the other views the potential is untouched, so re-derivation is a
    # no-op and the run records that it was.
    for view in ("i", "r", "s"):
        a = sf.rotate_view(m, h, mu, P, 1.1, view, coherent=True)
        b = sf.rotate_view(m, h, mu, P, 1.1, view, coherent=False)
        assert torch.equal(a, b)
        assert not sf.coherent_is_effective(view, True)


def test_family_score_separates_a_shift_from_a_turn():
    rng = np.random.default_rng(0)
    base = torch.as_tensor(rng.normal(size=(200, 8)), dtype=torch.float32)
    shift = torch.as_tensor(rng.normal(size=(1, 8)), dtype=torch.float32)
    translation = sf.family_score(base + shift, base - shift)
    angle = torch.as_tensor(rng.uniform(-np.pi, np.pi, 200), dtype=torch.float32)
    turn = torch.zeros(200, 8)
    turn[:, 0], turn[:, 1] = torch.cos(angle), torch.sin(angle)
    rotation = sf.family_score(turn, -turn)
    assert translation["translation_score"] > rotation["translation_score"]


def test_cached_rollout_dispatches_to_the_spiking_core():
    for cls in (SpikingArmWM, RateArmWM):
        m = cls(neurons=6)
        m.eval()
        rows = torch.rand(4, 3, 3 * 16 * 32 + 1 + 2)
        rows[..., 3 * 16 * 32] = 1.0
        cache = sf.SpikeCachedRollout(m, rows)
        h = m.initial_state(4, rows)
        out = cache.roll(h, joint=0, delta=0.2, want_pred=False)
        assert out.shape == (4, m.hidden)
        preds = cache.roll(h, joint=0, delta=0.0, want_pred=True)
        assert preds.shape == (3, 4, 3 * 16 * 32)
