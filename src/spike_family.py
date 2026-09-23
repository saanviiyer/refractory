"""Which one-parameter group does each actuator induce, on each state view?

The parent's family detector fits a rotation and a translation on the same
targets at the same scale and lets the lower residual decide.  It was validated
in both directions: on freely turning joints it says rotation 8/8, on joints
with stops it says translation 8/8, with renderer, image size, colours, sensor
noise, action space, architecture and budget held fixed.

A spiking core makes the question sharper rather than only repeating it.  The
membrane potential is an integrator, the spike vector is a binary event code,
and the trace is a low-pass filter of the events.  There is no reason in
advance why an action that turns one of them should turn the others, and if the
family verdict changes across views then "the actuator induces a rotation" is a
statement about the analyst's choice of state variable.

Bounded joints are where this matters most, because a saturating actuator is
also the case where the parent's rotation-only criterion returned nothing.
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch

import arm_env
import spike_fast as sf
import train_arm_visual as TAV
from analyze_arm_zeroshot import DELTAS, collect_states
from arm_family import TRANSLATION_GRID, _refine
from arm_torus import COARSE, plane_from_diff
from gate_blackout import require_pass
from spiking import VIEW_ORDER, build_model, calibrate_threshold
from surrogate_diag import degenerate_tags

VIEWS = VIEW_ORDER + ("all",)


def fit_families(cache, h, mu, joint, view, coherent, probe=0.4):
    """Fit rotation and translation on the same targets within one view."""
    plus, minus = sf.signed_response(cache, h, joint, view, probe)
    family = sf.family_score(plus, minus)
    direction = np.asarray(family["direction"], dtype=np.float64)
    plane, variance = plane_from_diff(torch.cat([minus, plus]).numpy())

    def rot_at(g):
        return sf.conjugacy_losses(cache, h, mu, plane, joint, DELTAS, g,
                                   arm_env.DT_ANGLE, view, coherent=coherent)

    def tra_at(g):
        return sf.translation_losses(cache, h, direction, joint, DELTAS, g,
                                     arm_env.DT_ANGLE, view, coherent=coherent)

    rot = _refine(rot_at(COARSE), COARSE, rot_at)
    tra = _refine(tra_at(TRANSLATION_GRID), TRANSLATION_GRID, tra_at)
    winner = "rotation" if rot["residual"] < tra["residual"] else "translation"
    return {"joint": joint + 1, "view": view,
            "translation_score": family["translation_score"],
            "response_mean_norm": family["mean_norm"],
            "response_spread": family["residual_spread"],
            "rotation": {k: (round(v, 4) if isinstance(v, float) else v)
                         for k, v in rot.items()},
            "translation": {k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in tra.items()},
            "family": winner,
            "margin": round(abs(rot["residual"] - tra["residual"]), 4),
            "response_var_frac": variance}


def analyse(path: Path, batch=128, variant="v7", probe=0.4, views=VIEWS,
            coherent=False, untrained=False):
    tag = path.stem
    _, model_kind, cond, seed_text = tag.split("_")
    seed = int(seed_text[1:])
    base = build_model(model_kind)
    if untrained:
        # PREREG A5.  Random weights, no gradient step, but the same
        # pre-gradient threshold calibration the trainer applies: without it
        # the population never fires and the control returns "nothing fits" by
        # construction, which is an artefact wearing the reassuring answer's
        # clothes.
        torch.manual_seed(90_000 + seed)
        base = build_model(model_kind)
        rng = np.random.default_rng(seed + 1_777)
        warm = TAV.make_batch(16, 32, cond, seed, rng, variant=variant)
        calibrate_threshold(base, warm[0], warm[2], warm[3])
    else:
        base.load_state_dict(torch.load(path, map_location="cpu"))
    base.eval()
    t0 = time.time()
    h, rows, _ = collect_states(base, 61_001, batch=batch, variant=variant)
    mu = h.mean(0, keepdim=True)
    cache = sf.SpikeCachedRollout(base, rows)
    out = [fit_families(cache, h, mu, j, v, coherent, probe)
           for v in views for j in (0, 1)]
    return {"tag": tag, "model_kind": model_kind, "cond": cond,
            "seed": seed, "variant": variant,
            "coherent": bool(coherent), "untrained": bool(untrained),
            "discovery_uses_joint_labels": False,
            "state_dim": int(base.hidden), "fits": out,
            "seconds": round(time.time() - t0, 1)}


def main(indir, output, gate, batch, variant, conds, models, views, coherent,
         allow_partial=False, diag=None, untrained=False):
    allow = ("PASS", "PARTIAL") if allow_partial else ("PASS",)
    if untrained:
        # A5: the gate is bypassed on the record.  An untrained model will not
        # beat frame persistence, and refusing to analyse it would defeat the
        # control, whose whole point is what the criterion reports on a model
        # that has learned nothing.
        gate_report = {"verdict": "bypassed (untrained control, PREREG A5)"}
        print("gate bypassed: untrained control (PREREG A5)", flush=True)
    else:
        gate_report = require_pass(gate, allow)
    excluded = degenerate_tags(diag)
    rows, skipped = [], []
    for filename in sorted(glob.glob(str(indir / "arm_*_*.pt"))):
        stem = Path(filename).stem.split("_")
        if stem[1] not in models or stem[2] not in conds:
            continue
        if Path(filename).stem in excluded:
            skipped.append(Path(filename).stem)
            print("excluded (degenerate population)", Path(filename).stem,
                  flush=True)
            continue
        row = analyse(Path(filename), batch, variant, views=views,
                      coherent=coherent, untrained=untrained)
        rows.append(row)
        print(row["tag"], [(f["view"], f["joint"], f["family"],
                            f["rotation"]["residual"],
                            f["translation"]["residual"])
                           for f in row["fits"]],
              f"{row['seconds']}s", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"variant": variant, "views": list(views),
                                  "coherent": bool(coherent),
                                  "untrained": bool(untrained),
                                  "gate": str(gate),
                                  "gate_verdict": gate_report.get("verdict"),
                                  "allow_partial": bool(allow_partial),
                                  "excluded_degenerate": skipped,
                                  "rows": rows}, indent=2))
    print(output)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=Path, default=Path("runs/snn_v7"))
    ap.add_argument("--output", type=Path,
                    default=Path("runs/spike_family_v7.json"))
    ap.add_argument("--gate", type=Path, default=Path("runs/gate_snn_v7.json"))
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--variant", default="v7")
    ap.add_argument("--conds", default="dark")
    ap.add_argument("--models", default="snn,rate")
    ap.add_argument("--views", default=",".join(VIEWS))
    ap.add_argument("--coherent", action="store_true")
    ap.add_argument("--allow-partial", action="store_true",
                    help="declared deviation: analyse a PARTIAL gate verdict")
    ap.add_argument("--diag", type=Path, default=None)
    ap.add_argument("--untrained", action="store_true",
                    help="PREREG A5: random init, calibrated, gate bypassed")
    a = ap.parse_args()
    main(a.indir, a.output, a.gate, a.batch, a.variant,
         set(a.conds.split(",")), set(a.models.split(",")),
         tuple(a.views.split(",")), a.coherent, a.allow_partial, a.diag,
         a.untrained)
