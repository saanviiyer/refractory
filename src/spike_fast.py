"""Cached rollout and view-restricted transforms for spiking world models.

Two changes from ``vendor.arm_fast``, and nothing else.

The core advances a flat state made of four stacked components rather than one
GRU hidden vector, so ``_advance`` dispatches to the model's own ``advance``.

The candidate transform acts on a **view**: a named contiguous block of that
flat state.  The parent rotates the whole hidden vector because there is only
one, so the parent's transform is the ``view="all"`` case of this one, and
``tests/test_spike_fast.py`` checks that the two agree numerically on a model
whose state is a single block.

``coherent`` mode additionally re-derives the emitted spike vector from the
rotated membrane potential.  It answers a different question from the default:
view-only asks what happens if this component is moved, coherent asks what
happens if the neuron's state is moved.  Re-derivation is only defined when the
membrane potential is among the rotated coordinates, so for the other views the
two modes coincide and the result records that they did.
"""
from __future__ import annotations

import numpy as np
import torch

from arm_fast import CachedRollout

COHERENT_APPLIES = ("v", "all")


class SpikeCachedRollout(CachedRollout):
    """``CachedRollout`` for the spiking core.  Nothing else is changed."""

    def _advance(self, z, mask, action, h):
        if self.kind in ("spiking_arm", "rate_arm", "graded_rate_arm"):
            return self.base.advance(z, mask, action, h)
        return super()._advance(z, mask, action, h)


def view_slices(model):
    """Named blocks of the flat state, plus the whole state under ``all``."""
    out = dict(model.views())
    out["all"] = slice(0, model.hidden)
    return out


def _recohere(model, h):
    """Re-derive the spike vector from the (possibly rotated) potential."""
    v = h[..., model.view_slice("v")]
    s = model._emit(v - model.threshold)
    h = h.clone()
    h[..., model.view_slice("s")] = s
    return h


def rotate_view(model, h, mu, plane, phi, view, coherent=False):
    """Rotate ``view``'s coordinates of ``h`` by ``phi``; leave the rest alone.

    ``phi`` may be a scalar or one angle per state, which is what the
    clamp-corrected fit needs: the actuator saturates at +-1, so every state
    must be rotated by the angle its own effective perturbation implies.
    """
    sl = view_slices(model)[view]
    plane = torch.as_tensor(np.asarray(plane), dtype=h.dtype)
    block, mu_block = h[..., sl], mu[..., sl].to(h.dtype)
    c = (block - mu_block) @ plane
    phi = torch.as_tensor(phi, dtype=h.dtype)
    if phi.ndim == 1:
        phi = phi[:, None]
    ca, sa = torch.cos(phi), torch.sin(phi)
    ca = ca.reshape(-1) if ca.numel() > 1 else ca.reshape(())
    sa = sa.reshape(-1) if sa.numel() > 1 else sa.reshape(())
    c2 = torch.stack([ca * c[:, 0] - sa * c[:, 1],
                      sa * c[:, 0] + ca * c[:, 1]], 1)
    out = h.clone()
    out[..., sl] = block + (c2 - c) @ plane.T
    if coherent and view in COHERENT_APPLIES:
        out = _recohere(model, out)
    return out


def translate_view(model, h, direction, shift, view, coherent=False):
    """Slide ``view``'s coordinates of ``h`` along ``direction``.

    The translation counterpart of ``rotate_view``.  A bounded joint is carried
    by a direction the action slides the state along, not by a plane it turns
    the state within, so the two families must be fitted on the same targets at
    the same scale before either is quoted.
    """
    sl = view_slices(model)[view]
    direction = torch.as_tensor(np.asarray(direction),
                                dtype=h.dtype).reshape(1, -1)
    shift = torch.as_tensor(shift, dtype=h.dtype)
    shift = shift.reshape(1, 1) if shift.ndim == 0 else shift.reshape(-1, 1)
    out = h.clone()
    out[..., sl] = h[..., sl] + shift * direction
    if coherent and view in COHERENT_APPLIES:
        out = _recohere(model, out)
    return out


def coherent_is_effective(view, coherent):
    """Did ``coherent`` actually change anything for this view?"""
    return bool(coherent and view in COHERENT_APPLIES)


def response_diff(cache, h, joint, view, probe=0.4):
    """State displacement in ``view`` caused by perturbing one actuator.

    No labels.  The perturbation is applied to the action, never to the state,
    so this is the parent's discovery signal restricted to a block.
    """
    sl = view_slices(cache.base)[view]
    base = cache.roll(h, joint, 0.0, want_pred=False)
    plus = cache.roll(h, joint, probe, want_pred=False)
    minus = cache.roll(h, joint, -probe, want_pred=False)
    return torch.cat([minus[..., sl] - base[..., sl],
                      plus[..., sl] - base[..., sl]])


def signed_response(cache, h, joint, view, probe=0.4):
    """The +/- responses kept apart, for the rotation-vs-translation score."""
    sl = view_slices(cache.base)[view]
    base = cache.roll(h, joint, 0.0, want_pred=False)
    plus = cache.roll(h, joint, probe, want_pred=False)[..., sl] - base[..., sl]
    minus = cache.roll(h, joint, -probe, want_pred=False)[..., sl] - base[..., sl]
    return plus, minus


def _losses(cache, h, deltas, gains, joint, make_state, chunk=6):
    """Shared inner loop: prediction error of a transformed start state.

    The target for perturbation ``d`` is the frozen model's own rollout with
    actuator ``joint`` perturbed by ``d``.  The candidate is the unperturbed
    rollout started from a state the candidate transform has moved.  Include
    0.0 in ``gains`` to obtain the no-transform reference.
    """
    targets = torch.stack([cache.roll(h, joint, float(d)) for d in deltas], 1)
    T, D, B, P = targets.shape
    flat = targets.reshape(T, D * B, P)
    gains = np.asarray(gains, dtype=np.float64)
    out = np.empty(len(gains), dtype=np.float64)
    for start in range(0, len(gains), chunk):
        block = gains[start:start + chunk]
        states = [make_state(float(g), float(d)) for g in block for d in deltas]
        H = torch.cat(states, 0)
        rep = len(block) * D
        total = torch.zeros(len(block), dtype=torch.float64)
        for t in range(cache.T):
            z, mask, action = cache._inputs(t, rep)
            H = cache._advance(z, mask, action, H)
            pred = cache.base.readout(H).reshape(rep * B, -1)
            err = (pred - flat[t].repeat(len(block), 1)) ** 2
            total += err.reshape(len(block), D * B * P).mean(1).double()
        out[start:start + len(block)] = (total / cache.T).numpy()
    return out


@torch.no_grad()
def conjugacy_losses(cache, h, mu, plane, joint, deltas, gains, dt, view,
                     chunk=6, clamp_corrected=True, coherent=False):
    """Rotation family: does turning ``view`` in ``plane`` imitate the action?"""
    model = cache.base

    def make(g, d):
        if clamp_corrected:
            phi = (g * dt) * cache.effective_delta(joint, d)
        else:
            phi = torch.tensor(g * d * dt, dtype=h.dtype)
        return rotate_view(model, h, mu, plane, phi, view, coherent)

    return _losses(cache, h, deltas, gains, joint, make, chunk)


@torch.no_grad()
def translation_losses(cache, h, direction, joint, deltas, gains, dt, view,
                       chunk=6, clamp_corrected=True, coherent=False):
    """Translation family: does sliding ``view`` along ``direction`` imitate it?

    Identical in structure to ``conjugacy_losses`` and fitted on the same
    targets, so the two residuals are directly comparable.
    """
    model = cache.base

    def make(g, d):
        if clamp_corrected:
            shift = (g * dt) * cache.effective_delta(joint, d)
        else:
            shift = torch.tensor(g * d * dt, dtype=h.dtype)
        return translate_view(model, h, direction, shift, view, coherent)

    return _losses(cache, h, deltas, gains, joint, make, chunk)


def family_score(plus, minus):
    """Cheap hint, not the verdict: is this response a rotation or a shift?

    A translation moves every state by the same vector, so the signed mean is
    large relative to its spread.  A rotation moves each state according to
    where it sits on the circle, so the signed mean cancels.  The parent's
    validation found the two families separate a hundredfold on synthetic
    responses and about twofold on real ones; the residual comparison decides.
    """
    signed = (plus - minus) / 2.0
    mean = signed.mean(0)
    spread = (signed - mean).pow(2).sum(1).mean().sqrt()
    return {"mean_norm": round(float(mean.norm()), 5),
            "residual_spread": round(float(spread), 5),
            "translation_score": round(
                float(mean.norm() / (spread + 1e-12)), 5),
            "direction": (mean / (mean.norm() + 1e-12)).tolist()}
