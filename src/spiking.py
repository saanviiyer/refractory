"""Spiking recurrent world models for the two-joint arm, plus their rate twin.

The parent project's criterion asks whether rotating the state inside a plane
imitates taking an extra action.  It has been demonstrated on one architecture,
a convolutional GRU, because the only other architecture tried (an RSSM) never
cleared the validity gate.  A spiking core is a second architecture in a
stronger sense than another gated RNN would be: the recurrent signal is a
binary event, the state is reset rather than blended, and there is no single
vector that is obviously "the latent state".

That last point is the reason this project exists rather than being a port.  A
GRU has one hidden vector, so "which coordinates do the dynamics use" is asked
of it and there is nothing to choose.  A spiking core carries four state
components of equal standing --- membrane potential, synaptic current, spike
trace, and the emitted spike vector --- and a probing paper would treat any of
them as "the representation".  If the criterion returns a different verdict
depending on which one it is pointed at, then "the dynamics use this
coordinate" is under-specified in the same way "the variable is decodable" was.

Encoder and decoder are byte-identical in shape to ``vendor.visual_models``'s
so that nothing but the recurrent core differs.

Neuron model, current-based leaky integrate-and-fire with reset to zero::

    i[t] = alpha * i[t-1] + W_in x[t] + W_rec s[t-1]
    v[t] = beta * v[t-1] * (1 - s[t-1]) + (1 - beta) * i[t]
    s[t] = Theta(v[t] - v_th)
    r[t] = gamma * r[t-1] + (1 - gamma) * s[t]

The decoder reads ``r``.  ``s`` is a deterministic function of ``v``; it is
still exposed as a view because asking whether rotating the *spike pattern*
imitates an action is a different question from asking it of ``v``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from visual_env import IMAGE_H, IMAGE_W

# Order fixes the layout of the flat state vector and must not be permuted:
# checkpoints, cached rollouts and every recorded view slice depend on it.
VIEW_ORDER = ("v", "i", "r", "s")
# The state the decoder reads, and so the closest analogue of the GRU hidden
# vector.  Pre-registered as the primary view in PREREG.md.
PRIMARY_VIEW = "r"

SURROGATE_WIDTH = 4.0
# PREREG A1.  The first twin's emission saturates at the margins the model
# learns -- sigmoid(4 * 1.9) is 0.9995, a spike in all but name -- so it shares
# the property it was built to remove.  A tenth of the width keeps the emission
# graded across the observed margin range.
EMISSION_WIDTH_SOFT = 0.25


class _SpikeFn(torch.autograd.Function):
    """Heaviside forward, fast-sigmoid surrogate backward.

    The backward is ``1 / (1 + width * |x|)^2``, which is the derivative of a
    fast sigmoid of scale ``1 / width``.  ``width`` enters the graph as a saved
    tensor rather than a closure so a checkpoint records the value it was
    trained with.
    """

    @staticmethod
    def forward(ctx, x, width):
        ctx.save_for_backward(x, width)
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        x, width = ctx.saved_tensors
        return grad_out / (1.0 + width * x.abs()) ** 2, None


def spike(x, width):
    return _SpikeFn.apply(x, torch.as_tensor(width, dtype=x.dtype))


def soft_spike(x, width):
    """The surrogate's own forward function: what the backward pass assumes.

    The rate twin uses this in place of the Heaviside.  Parameter count,
    initialisation, optimiser and gradient *form* are then identical to the
    spiking model, so a difference between the two is attributable to the
    discreteness of the emitted signal and not to the architecture change.
    """
    return torch.sigmoid(width * x)


def _encoder():
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(),
        nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
        nn.Conv2d(32, 32, 3, 2, 1), nn.ReLU(), nn.Flatten(),
        nn.Linear(32 * (IMAGE_H // 8) * (IMAGE_W // 8), 96), nn.ReLU())


def _decoder(width):
    return nn.Sequential(nn.Linear(width, 256), nn.ReLU(),
                         nn.Linear(256, 3 * IMAGE_H * IMAGE_W), nn.Sigmoid())


class SpikingArmWM(nn.Module):
    """Pixel-predictive world model with a spiking recurrent core.

    ``neurons=128`` matches the GRU's hidden width, so every view has the same
    dimension as the GRU state the criterion was calibrated on and the
    variance-matched random-plane null is drawn from a space of the same size.
    ``neurons=250`` instead matches the GRU's parameter count.  Both are run;
    the width-matched one is primary, because the criterion's geometry is
    dimension-sensitive and its budget is not.
    """
    kind = "spiking_arm"
    discrete = True

    def __init__(self, neurons=128, surrogate_width=SURROGATE_WIDTH):
        super().__init__()
        self.neurons = neurons
        self.hidden = len(VIEW_ORDER) * neurons
        self.surrogate_width = float(surrogate_width)
        self.frame_encoder = _encoder()
        self.input = nn.Linear(96 + 1 + 2, neurons)
        self.recurrent = nn.Linear(neurons, neurons, bias=False)
        # Time constants are stored in logit space so decays stay in (0, 1)
        # without a clamp.  A clamp that always binds is a silent zero-gradient
        # path, which is the failure this line of work has already paid for.
        self.logit_alpha = nn.Parameter(torch.full((neurons,), 0.85))
        self.logit_beta = nn.Parameter(torch.full((neurons,), 1.40))
        self.logit_gamma = nn.Parameter(torch.full((neurons,), 0.40))
        self.threshold = nn.Parameter(torch.ones(neurons))
        self.decoder = _decoder(neurons)
        nn.init.orthogonal_(self.recurrent.weight, gain=0.6)

    # -- state layout ----------------------------------------------------
    def view_slice(self, name):
        return slice(VIEW_ORDER.index(name) * self.neurons,
                     (VIEW_ORDER.index(name) + 1) * self.neurons)

    def views(self):
        return {n: self.view_slice(n) for n in VIEW_ORDER}

    def split(self, h):
        return [h[..., self.view_slice(n)] for n in VIEW_ORDER]

    def initial_state(self, batch, ref):
        return ref.new_zeros(batch, self.hidden)

    # -- dynamics --------------------------------------------------------
    def _emit(self, margin):
        return spike(margin, self.surrogate_width)

    def advance(self, z, mask, action, h):
        """One step of the core from the flat state ``h``.

        ``z`` is the frame embedding, ``mask`` the blackout flag, ``action``
        the two normalised joint velocities.
        """
        v, i, r, s = self.split(h)
        alpha = torch.sigmoid(self.logit_alpha)
        beta = torch.sigmoid(self.logit_beta)
        gamma = torch.sigmoid(self.logit_gamma)
        drive = self.input(torch.cat([z, mask[:, None], action], 1)) \
            + self.recurrent(s)
        i = alpha * i + drive
        v = beta * v * (1.0 - s) + (1.0 - beta) * i
        s = self._emit(v - self.threshold)
        r = gamma * r + (1.0 - gamma) * s
        return torch.cat([v, i, r, s], dim=-1)

    def step(self, frame, mask, action, h):
        return self.advance(self.frame_encoder(frame), mask, action, h)

    def readout(self, h):
        """Decode without advancing the dynamics.  Reads the ``r`` view."""
        return self.decoder(h[..., self.view_slice(PRIMARY_VIEW)]) \
            .reshape(-1, 3, IMAGE_H, IMAGE_W)

    def forward(self, frames, mask, action, h0=None, return_h=False):
        B, T = frames.shape[:2]
        z = self.frame_encoder(frames.reshape(B * T, 3, IMAGE_H, IMAGE_W))
        z = z.reshape(B, T, -1)
        h = self.initial_state(B, z) if h0 is None else h0
        states = []
        for t in range(T):
            h = self.advance(z[:, t], mask[:, t], action[:, t], h)
            states.append(h)
        hidden = torch.stack(states, 1)
        pred = self.decoder(hidden[..., self.view_slice(PRIMARY_VIEW)]) \
            .reshape(B, T, 3, IMAGE_H, IMAGE_W)
        return (pred, hidden) if return_h else pred

    # -- instrumentation -------------------------------------------------
    @torch.no_grad()
    def margins(self, frames, mask, action):
        """Raw pre-threshold margins ``v - v_th``, one per neuron per step.

        Returned unclamped and unsummarised on purpose.  A surrogate gradient
        whose support does not reach the operating margin contributes nothing
        and the model still trains, converges and produces a number, exactly as
        a clamped regulariser does.  Nothing here is a gate; see
        ``gate_blackout.py`` for the gate.
        """
        B, T = frames.shape[:2]
        z = self.frame_encoder(frames.reshape(B * T, 3, IMAGE_H, IMAGE_W))
        z = z.reshape(B, T, -1)
        h = self.initial_state(B, z)
        out = []
        for t in range(T):
            v = self.split(h)[0]
            h = self.advance(z[:, t], mask[:, t], action[:, t], h)
            out.append(self.split(h)[0] - self.threshold)
            del v
        return torch.stack(out, 1)


class RateArmWM(SpikingArmWM):
    """The rate twin: same everything, continuous emission.

    This is the control that decides what any spiking result is about.  If the
    twin and the spiking model give the same verdict, nothing in the result is
    attributable to spikes.

    Measured on the v6 pilot, it is a *partial* control: the emission is graded
    on only about a quarter of steps, because the sigmoid saturates at the
    margins training produces.  See ``GradedRateArmWM`` and PREREG A1.
    """
    kind = "rate_arm"
    discrete = False

    def _emit(self, margin):
        return soft_spike(margin, self.surrogate_width)


class GradedRateArmWM(SpikingArmWM):
    """The graded control: no discreteness, no surrogate, no weak gradient.

    PREREG A1.  Relative to the spiking model this removes three things at once
    -- the step forward, the mismatch between forward and backward, and an
    emission gradient running at a few per cent of peak -- so it does not
    isolate discreteness.  It separates **spiking machinery** from
    **architecture**, which is the question the v6 pilot left open:

    - it fails  -> the four-component LIF architecture is what fails, and
                   nothing about spikes or surrogates explains it;
    - it passes -> the spiking machinery is implicated, and which part is a
                   further experiment.

    The emission is an ordinary sigmoid differentiated exactly by autograd.
    Parameter count, initialisation and every other component are unchanged, so
    it remains budget-matched to the models it is compared against.
    """
    kind = "graded_rate_arm"
    discrete = False

    def __init__(self, neurons=128, surrogate_width=SURROGATE_WIDTH,
                 emission_width=EMISSION_WIDTH_SOFT):
        super().__init__(neurons, surrogate_width)
        self.emission_width = float(emission_width)

    def _emit(self, margin):
        # No custom autograd: the backward pass is the true derivative of this
        # forward, which is the point.
        return torch.sigmoid(self.emission_width * margin)


@torch.no_grad()
def calibrate_threshold(model, frames, mask, action, quantile=0.90):
    """Set thresholds so the population starts inside the surrogate's support.

    A surrogate gradient is not a clamp and has no hard cutoff, but a
    population sitting many multiples of ``1/width`` away from threshold trains
    on a gradient that is numerically nothing, and nothing about the loss curve
    says so.  Left at a fixed 1.0 against this encoder's scale the untrained
    membrane distribution has standard deviation about 0.16 and no neuron ever
    fires, so the recurrent term is dead from the first step.

    One pass at zero threshold, then each neuron's threshold is set to a high
    quantile of its own potential.  This uses frames and actions only; no pose
    label is involved, and it happens before the first gradient step.
    """
    original = model.threshold.detach().clone()
    model.threshold.zero_()
    v = model.margins(frames, mask, action)          # threshold is 0, so v
    q = torch.quantile(v.reshape(-1, model.neurons).double(),
                       quantile, dim=0).to(original.dtype)
    # A neuron whose potential never varies gets its old threshold back rather
    # than a degenerate one.
    flat = v.reshape(-1, model.neurons)
    dead = flat.std(0) < 1e-6
    model.threshold.copy_(torch.where(dead, original, q))
    return {"quantile": quantile,
            "threshold_p10": round(float(model.threshold.quantile(0.10)), 5),
            "threshold_p50": round(float(model.threshold.median()), 5),
            "threshold_p90": round(float(model.threshold.quantile(0.90)), 5),
            "n_degenerate_neurons": int(dead.sum())}


BUILDERS = {
    # 211 neurons matches the GRU's 554,720 parameters to within 0.04%; 128
    # matches its hidden width instead, and so the dimension every view is
    # measured in.  Neither is a fair comparison alone, so both are run.
    "snn": lambda: SpikingArmWM(neurons=128),
    "snn_wide": lambda: SpikingArmWM(neurons=211),
    "rate": lambda: RateArmWM(neurons=128),
    "rate_wide": lambda: RateArmWM(neurons=211),
    "ratesoft": lambda: GradedRateArmWM(neurons=128),
}


def build_model(kind):
    """Spiking kinds here; anything else falls through to the parent's registry."""
    if kind in BUILDERS:
        return BUILDERS[kind]()
    from arm_models import build_arm_model
    return build_arm_model(kind)


def n_params(model):
    return sum(p.numel() for p in model.parameters())
