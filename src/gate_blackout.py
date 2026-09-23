"""The validity gate, and the refusal that depends on it.

A world model that cannot predict the arm during a blackout has no latent worth
probing, and a coordinate verdict read off one means nothing.  The parent
learned this the expensive way: two successive free-nats floors silently
switched a KL term off, the prior went untrained, reconstruction loss looked
healthy throughout, and no statistic of the loss caught it.  What caught it was
a direct downstream measurement of the thing the term was supposed to buy.

A spiking model has its own version of the same trap.  A surrogate gradient
whose support does not reach the neuron's operating margin contributes nothing;
the model still trains, still converges, and still produces a number.  The
lesson from the parent is that no statistic of the surrogate is allowed to be
the gate --- ``surrogate_diag.py`` reports those, and reports them as
diagnostics.  The gate is this file, and it measures prediction.

The comparator is frame persistence: during a blackout the model sees nothing,
so the cheapest non-trivial prediction is "the scene has not moved since the
last frame I saw".  ``check`` and ``_last_visible`` are the parent's, imported
rather than reimplemented, so a verdict here is on the same scale as the
parent's published GRU and RSSM verdicts.

Calibrate before trusting.  ``--calibrate`` runs the gate against the parent's
own checkpoints, where the answer is already known: the conv GRU passes 8/8 and
the RSSM fails 0/8 on both v6 and v7.  A gate that does not reproduce that is
broken, and its verdict on a new architecture is worthless either way.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

import blackout_validity as bv
from spiking import build_model

# What the parent's runs report, and therefore what a working gate must say.
CALIBRATION = {("arm_v6", "gru"): "PASS", ("arm_v7", "gru"): "PASS",
               ("rssm_v6", "rssm"): "FAIL", ("rssm_v7", "rssm"): "FAIL"}


class SpikingArmSource(bv.ArmSource):
    """The parent's arm source with the spiking kinds added to its registry."""

    def model(self, kind="snn"):
        return build_model(kind)


def run(indir, variant, pattern, batch, T, seed):
    src = SpikingArmSource(variant)
    rows = []
    for f in sorted(glob.glob(str(Path(indir) / pattern))):
        r = bv.check(Path(f), src, batch, T, seed)
        rows.append(r)
        print(r["tag"], "model", r.get("blackout_mse_model"),
              "persistence", r.get("blackout_mse_persistence"),
              "ratio", r.get("ratio_model_over_persistence"),
              "PASS" if r.get("beats_persistence") else "FAIL", flush=True)
    n_pass = sum(1 for r in rows if r.get("beats_persistence"))
    verdict = "PASS" if rows and n_pass == len(rows) else (
        "PARTIAL" if n_pass else "FAIL")
    ratios = [r["ratio_model_over_persistence"] for r in rows
              if r.get("ratio_model_over_persistence") is not None]
    return {"gate": "blackout_prediction_beats_frame_persistence",
            "comparator": "frame persistence on blacked-out steps",
            "domain": "arm", "task": variant, "indir": str(indir),
            "pattern": pattern, "n_models": len(rows),
            "n_beating_persistence": n_pass, "verdict": verdict,
            "median_ratio": round(float(np.median(ratios)), 4)
            if ratios else None, "rows": rows}


def require_pass(path, allow=("PASS",)):
    """Refuse to read an analysis whose checkpoints did not clear the gate.

    Raises rather than warns.  A warning is a thing that gets scrolled past,
    and the whole point of the gate is that the failure it catches looks
    perfectly healthy from every other angle.
    """
    path = Path(path)
    if not path.exists():
        raise SystemExit(
            f"validity gate {path} has not been run. Run gate_blackout.py "
            "before any coordinate analysis; results read off an ungated "
            "checkpoint family are not interpretable.")
    report = json.loads(path.read_text())
    if report.get("verdict") not in allow:
        raise SystemExit(
            f"validity gate {path} reports {report.get('verdict')} "
            f"({report.get('n_beating_persistence')}/{report.get('n_models')} "
            f"beat persistence, median ratio "
            f"{report.get('median_ratio')}). Analysis refused.")
    return report


def calibrate(parent_runs, batch, T, seed):
    """Check the gate passes a known-good family and fails a known-bad one."""
    parent_runs = Path(parent_runs)
    out = []
    for (subdir, kind), expected in CALIBRATION.items():
        indir = parent_runs / subdir
        if not indir.exists():
            out.append({"indir": str(indir), "expected": expected,
                        "got": None, "agrees": None, "note": "not present"})
            continue
        variant = "v6" if subdir.endswith("v6") else "v7"
        report = run(indir, variant, f"arm_{kind}_dark_s*.pt", batch, T, seed)
        out.append({"indir": str(indir), "kind": kind, "expected": expected,
                    "got": report["verdict"],
                    "n_beating_persistence": report["n_beating_persistence"],
                    "n_models": report["n_models"],
                    "median_ratio": report["median_ratio"],
                    "agrees": report["verdict"] == expected})
    checked = [r for r in out if r["agrees"] is not None]
    return {"calibration": "gate reproduces the parent's known verdicts",
            "n_checked": len(checked),
            "n_agreeing": sum(1 for r in checked if r["agrees"]),
            "calibrated": bool(checked) and all(r["agrees"] for r in checked),
            "rows": out}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="runs/snn_v6")
    ap.add_argument("--output", default="runs/gate_snn_v6.json")
    ap.add_argument("--variant", default="v6")
    ap.add_argument("--pattern", default="arm_*_dark_s*.pt")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--T", type=int, default=48)
    ap.add_argument("--seed", type=int, default=91_000)
    ap.add_argument("--calibrate", action="store_true",
                    help="run against the parent's checkpoints instead")
    ap.add_argument("--parent-runs",
                    default="../used_coordinates/runs")
    a = ap.parse_args()
    report = (calibrate(a.parent_runs, a.batch, a.T, a.seed) if a.calibrate
              else run(a.indir, a.variant, a.pattern, a.batch, a.T, a.seed))
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(report, indent=2))
    if a.calibrate:
        print(f"calibrated={report['calibrated']} "
              f"{report['n_agreeing']}/{report['n_checked']} -> {a.output}")
    else:
        print(f"{report['verdict']}: {report['n_beating_persistence']}/"
              f"{report['n_models']} beat persistence -> {a.output}")
