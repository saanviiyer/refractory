"""Raw pre-threshold margins.  A diagnostic, deliberately not a gate.

The spiking analogue of the parent's clamped regulariser is a surrogate
gradient whose support does not reach the margins the neurons actually operate
at.  The fast-sigmoid surrogate ``1 / (1 + w|x|)^2`` has no hard cutoff, but at
width ``w`` it has fallen to one per cent of its peak by ``|x| = 9/w``, so a
population sitting well outside that band trains on a gradient that is
numerically nothing.  Loss still falls, the run still converges, and the
checkpoint still produces a coordinate verdict.

The parent's lesson is not "add a better statistic of the term".  It is that no
statistic of the term can separate "never engaged" from "engaged and
converged", and that the only thing that can is a direct downstream measurement
of what the term is supposed to buy.  So this file reports the raw margins,
uncollapsed, and refuses to have a threshold.  ``gate_blackout.py`` is the gate.

Also reported: the fraction of neurons that never fire and the fraction that
fire at every step.  Both are dead in the same silent way, and both make a
coordinate verdict meaningless while leaving the loss curve intact.

And the fraction of emitted values that are actually graded, which is a
diagnostic of the *control* rather than of the model.  The rate twin isolates
discreteness only if its emission is genuinely continuous, and a saturating
sigmoid at a large learned margin is not: ``sigmoid(4 * 1.9)`` is 0.9995, which
is a spike in all but name.  A twin whose emission is mostly saturated shares
the property it was built to remove, and any "both fail, so it is not about
discreteness" reading has to be discounted by exactly this number.  Zero by
construction for the spiking model.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

import train_arm_visual as TAV
from spiking import build_model

# Where the fast-sigmoid surrogate has fallen to 1% of its peak.  Reported as a
# scale for reading the margin quantiles against, not as a pass mark.
ONE_PERCENT_MULTIPLE = 9.0

# PREREG.md item 2.  A population that never fires, or fires at every step,
# carries a state that cannot turn, so a coordinate verdict read off it is
# meaningless.  Applied to individual checkpoints; a cell losing more than two
# of eight seeds is reported as degenerate rather than analysed.
DEGENERATE_FRACTION = 0.5

# An emitted value outside this band is a spike in all but name.
GRADED_LO, GRADED_HI = 0.02, 0.98


def is_degenerate(row):
    return (row["frac_neurons_never_fire"] > DEGENERATE_FRACTION
            or row["frac_neurons_always_fire"] > DEGENERATE_FRACTION)


def degenerate_tags(path):
    """Tags to exclude, from a diagnostic report.  Empty if there is none."""
    path = Path(path) if path else None
    if path is None or not path.exists():
        return set()
    report = json.loads(path.read_text())
    return {r["tag"] for r in report["rows"] if r.get("degenerate")}


@torch.no_grad()
def margin_report(model, variant, seed, batch=16, T=32, cond="dark"):
    """Quantiles of ``v - v_th``, and how much of the population is inert."""
    rng = np.random.default_rng(seed + 5_101)
    masked, _, mask, action, _ = TAV.make_batch(batch, T, cond, seed, rng,
                                                variant=variant)
    margins = model.margins(masked, mask, action)
    flat = margins.reshape(-1).numpy()
    q = np.percentile(np.abs(flat), [10, 50, 90, 99])
    fired = (margins > 0).float()
    # What the model actually emits, which for the twin need not be graded.
    emitted = model._emit(margins)
    graded = ((emitted > GRADED_LO) & (emitted < GRADED_HI)).float().mean()
    per_neuron = fired.mean((0, 1))
    width = float(getattr(model, "surrogate_width", float("nan")))
    return {
        "abs_margin_p10": round(float(q[0]), 5),
        "abs_margin_p50": round(float(q[1]), 5),
        "abs_margin_p90": round(float(q[2]), 5),
        "abs_margin_p99": round(float(q[3]), 5),
        "surrogate_width": width,
        "surrogate_1pct_margin": round(ONE_PERCENT_MULTIPLE / width, 5)
        if width else None,
        "mean_firing_rate": round(float(fired.mean()), 5),
        "frac_emission_graded": round(float(graded), 5),
        "frac_neurons_never_fire": round(float((per_neuron == 0).float()
                                               .mean()), 5),
        "frac_neurons_always_fire": round(float((per_neuron == 1).float()
                                                .mean()), 5),
        "note": "diagnostic only; the gate is gate_blackout.py",
    }


def main(indir, output, variant, pattern):
    rows = []
    for f in sorted(glob.glob(str(Path(indir) / pattern))):
        path = Path(f)
        kind = path.stem.split("_")[1]
        model = build_model(kind)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        seed = int(path.stem.split("_s")[-1])
        row = {"tag": path.stem, "model_kind": kind, "seed": seed,
               **margin_report(model, variant, seed)}
        row["degenerate"] = is_degenerate(row)
        rows.append(row)
        print(row["tag"], "p50", row["abs_margin_p50"], "1%@",
              row["surrogate_1pct_margin"], "rate", row["mean_firing_rate"],
              "dead", row["frac_neurons_never_fire"],
              "DEGENERATE" if row["degenerate"] else "", flush=True)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(
        {"diagnostic": "raw pre-threshold margins", "is_gate": False,
         "exclusion_rule": "PREREG.md item 2",
         "degenerate_fraction": DEGENERATE_FRACTION,
         "n_degenerate": sum(r["degenerate"] for r in rows),
         "variant": variant, "rows": rows}, indent=2))
    print(output)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="runs/snn_v6")
    ap.add_argument("--output", default="runs/surrogate_diag_v6.json")
    ap.add_argument("--variant", default="v6")
    ap.add_argument("--pattern", default="arm_*_dark_s*.pt")
    a = ap.parse_args()
    main(a.indir, a.output, a.variant, a.pattern)
