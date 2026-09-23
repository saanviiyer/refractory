"""The used-coordinates criterion, run once per state view of a spiking model.

Discovery uses actions and the frozen model's own predictions.  No pose labels,
no supervised decoder, no ground truth of any kind enters before scoring.  That
is the parent's rule and it is unchanged here.

What is new is the loop.  The parent asks its question of one hidden vector
because a GRU has one.  A spiking core carries four components of equal
standing, so the same question is asked of each, and the object of interest is
whether the answers agree.  Agreement means the criterion reads a property of
the model.  Disagreement means it reads a property of the state variable the
analyst chose, which would put "the dynamics use this coordinate" in the same
position the parent put "the variable is decodable".

The grids, deflation, search, matched-variance nulls, basis scoring and
uniformity check are the parent's, imported rather than reimplemented.
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
import interventions as iv
import zeroshot as zs
import spike_fast as sf
from analyze_arm_zeroshot import DELTAS, collect_states
from arm_torus import (BASES, COARSE, FINE_HALFWIDTH, FINE_POINTS, SEARCH_GRID,
                       deflate_matrix, plane_from_diff, subspace_from_diff,
                       uniformity)
from gate_blackout import require_pass
from spiking import VIEW_ORDER, build_model
from surrogate_diag import degenerate_tags

# The whole state is included so the single-block case the parent measures is
# recoverable from this project's own runs rather than only by comparison.
VIEWS = VIEW_ORDER + ("all",)


def search_plane(cache, h, mu, subspace, joint, view, coherent,
                 n_candidates=24, seed=0):
    """Best conjugate plane inside a response subspace, on one view.

    The top-two principal directions of a response are the directions it moves
    most, which need not be the plane the action rotates.  The principal plane
    is included as a candidate so the search cannot do worse than the default.
    """
    rng = np.random.default_rng(seed)
    k = subspace.shape[1]
    candidates = [subspace[:, :2]]
    for _ in range(n_candidates):
        q, _ = np.linalg.qr(rng.normal(size=(k, 2)))
        candidates.append(subspace @ q)
    best, best_cost = candidates[0], np.inf
    for plane in candidates:
        losses = sf.conjugacy_losses(cache, h, mu, plane, joint, DELTAS,
                                     SEARCH_GRID, arm_env.DT_ANGLE, view,
                                     coherent=coherent)
        still = float(losses[int(np.argmin(np.abs(SEARCH_GRID)))])
        cost = float(np.min(losses)) / (still + 1e-12)
        if cost < best_cost:
            best, best_cost = plane, cost
    return best, best_cost


def conjugacy(cache, h, mu, plane, joint, view, coherent):
    """Best rotation-imitates-action fit on ``plane`` within ``view``."""
    losses = sf.conjugacy_losses(cache, h, mu, plane, joint, DELTAS, COARSE,
                                 arm_env.DT_ANGLE, view, coherent=coherent)
    still = float(losses[int(np.argmin(np.abs(COARSE)))])
    best = int(np.argmin(losses))
    at_edge = best in (0, len(COARSE) - 1)
    fine = np.linspace(COARSE[best] - FINE_HALFWIDTH,
                       COARSE[best] + FINE_HALFWIDTH, FINE_POINTS)
    fine_losses = sf.conjugacy_losses(cache, h, mu, plane, joint, DELTAS, fine,
                                      arm_env.DT_ANGLE, view, coherent=coherent)
    b2 = int(np.argmin(fine_losses))
    return {"gain": float(fine[b2]),
            "residual": float(fine_losses[b2] / (still + 1e-12)),
            "gain_at_grid_edge": bool(at_edge)}


def phase_of(block, mu_block, plane):
    c = (np.asarray(block, dtype=np.float64)
         - np.asarray(mu_block, dtype=np.float64)) @ np.asarray(plane)
    return np.arctan2(c[:, 1], c[:, 0])


def phase_shift(cache, h, mu, plane, moving_joint, view, probe=0.4):
    """Median absolute phase movement on ``plane`` when one actuator acts.

    A torus needs two commuting circle actions: each action should turn its own
    circle and leave the other alone.  Radians.
    """
    sl = sf.view_slices(cache.base)[view]
    base = cache.roll(h, moving_joint, 0.0, want_pred=False)[..., sl].numpy()
    moved = cache.roll(h, moving_joint, probe, want_pred=False)[..., sl].numpy()
    mu_block = mu[..., sl].numpy()
    d = phase_of(moved, mu_block, plane) - phase_of(base, mu_block, plane)
    return float(np.median(np.abs(np.arctan2(np.sin(d), np.cos(d)))))


def score_bases(block, mu_block, plane, q_test):
    """Best integer combination a*q1 + b*q2 for this plane.  Evaluation only."""
    out = []
    for a, b in BASES:
        angle = arm_env.wrap(a * q_test[:, 0] + b * q_test[:, 1])
        s = zs.score_phase(block.numpy().astype(np.float64),
                           mu_block.numpy().astype(np.float64),
                           np.asarray(plane), angle)
        out.append({"basis": [a, b], "kappa": s["kappa"],
                    "winding": s["winding"]})
    out.sort(key=lambda r: -r["kappa"])
    return out


def discover(cache, h_fit, mu, order, mode, n_null, rng, view, coherent,
             probe=0.4):
    """Find one circle per actuator inside ``view``.  ``order`` fixes which first."""
    sl = sf.view_slices(cache.base)[view]
    dim = h_fit[..., sl].shape[1]
    diffs = {j: sf.response_diff(cache, h_fit, j, view, probe).numpy()
             for j in (0, 1)}
    magnitude = {j: float(np.median(np.linalg.norm(diffs[j], axis=1)))
                 for j in (0, 1)}
    planes, variances, taken, search_cost = {}, {}, None, {}
    for joint in order:
        projector = None
        if mode in ("deflate", "search") and taken is not None:
            projector = deflate_matrix(dim, taken)
        plane, variance = plane_from_diff(diffs[joint], projector)
        if mode == "search":
            subspace = subspace_from_diff(diffs[joint], projector)
            plane, cost = search_plane(cache, h_fit, mu, subspace, joint, view,
                                       coherent, seed=17 + joint)
            search_cost[joint] = round(float(cost), 4)
        planes[joint], variances[joint] = plane, variance
        taken = plane if taken is None else np.concatenate([taken, plane], 1)

    rows = []
    for joint in (0, 1):
        plane = planes[joint]
        own = conjugacy(cache, h_fit, mu, plane, joint, view, coherent)
        cross = conjugacy(cache, h_fit, mu, plane, 1 - joint, view, coherent)
        nulls = iv.matched_random_planes(h_fit[..., sl].numpy(),
                                         plane.astype(np.float32), rng,
                                         n_planes=n_null)
        nulls = nulls[0] if isinstance(nulls, tuple) else nulls
        null_res = [conjugacy(cache, h_fit, mu, p, joint, view,
                              coherent)["residual"] for p in nulls]
        plus, minus = sf.signed_response(cache, h_fit, joint, view, probe)
        rows.append({
            "joint": joint + 1,
            "response_var_frac": variances[joint],
            "response_magnitude": round(magnitude[joint], 5),
            "gain": round(own["gain"], 4),
            "residual": round(own["residual"], 4),
            "gain_at_grid_edge": own["gain_at_grid_edge"],
            "null_residual_median": round(float(np.median(null_res)), 4),
            "null_residual_min": round(float(np.min(null_res)), 4),
            "beats_null": bool(own["residual"] < float(np.median(null_res))),
            "cross_action_residual": round(cross["residual"], 4),
            "cross_action_gain": round(cross["gain"], 4),
            "self_phase_shift_rad": round(
                phase_shift(cache, h_fit, mu, plane, joint, view), 4),
            "crosstalk_phase_shift_rad": round(
                phase_shift(cache, h_fit, mu, plane, 1 - joint, view), 4),
            "search_cost": search_cost.get(joint),
            "family_hint": sf.family_score(plus, minus),
        })
    return planes, rows


def analyse_view(cache, h_fit, h_test, mu, q_test, view, coherent, mode, order,
                 n_null, rng):
    sl = sf.view_slices(cache.base)[view]
    planes, rows = discover(cache, h_fit, mu, order, mode, n_null, rng, view,
                            coherent)
    for r in rows:
        plane = planes[r["joint"] - 1]
        r["bases"] = score_bases(h_test[..., sl], mu[..., sl], plane, q_test)
        r["best_basis"] = r["bases"][0]["basis"]
        r["best_kappa"] = r["bases"][0]["kappa"]
        r["best_winding"] = r["bases"][0]["winding"]
        r["gain_over_winding"] = (round(r["gain"] / r["best_winding"], 3)
                                  if r["best_winding"] else None)
        a, b = r["best_basis"]
        r["uniformity"] = uniformity(
            h_test[..., sl], mu[..., sl], plane,
            arm_env.wrap(a * q_test[:, 0] + b * q_test[:, 1]))
    angles = iv.principal_angles(planes[0].astype(np.float32),
                                 planes[1].astype(np.float32))
    return {"view": view,
            "coherent_requested": bool(coherent),
            "coherent_effective": sf.coherent_is_effective(view, coherent),
            "view_dim": int(sl.stop - sl.start),
            "planes": rows,
            "plane_angles_deg": [round(float(x), 2) for x in angles]}


def analyse(path: Path, batch=128, n_null=4, variant="v6", mode="search",
            order=(0, 1), horizon=4, probe=0.4, deep=6, views=VIEWS,
            coherent=False, untrained=False):
    tag = path.stem
    _, model_kind, cond, seed_text = tag.split("_")
    seed = int(seed_text[1:])
    base = build_model(model_kind)
    if untrained:
        torch.manual_seed(90_000 + seed)
        base = build_model(model_kind)
    else:
        base.load_state_dict(torch.load(path, map_location="cpu"))
    base.eval()
    t0 = time.time()
    h_fit, rows_fit, _ = collect_states(base, 61_001, batch=batch,
                                        variant=variant, horizon=horizon,
                                        deep=deep)
    h_test, _, q_test = collect_states(base, 91_001, batch=batch,
                                       variant=variant, horizon=horizon,
                                       deep=deep)
    mu = h_fit.mean(0, keepdim=True)
    cache = sf.SpikeCachedRollout(base, rows_fit)
    out = []
    for view in views:
        rng = np.random.default_rng(31_000 + seed)
        out.append(analyse_view(cache, h_fit, h_test, mu, q_test, view,
                                coherent, mode, order, n_null, rng))
    return {"tag": tag, "model_kind": model_kind, "cond": cond, "seed": seed,
            "variant": variant, "mode": mode, "order": list(order),
            "horizon": horizon, "probe": probe, "deep": deep,
            "untrained": bool(untrained), "coherent": bool(coherent),
            "discovery_uses_joint_labels": False,
            "scoring_uses_joint_labels": True,
            "state_dim": int(base.hidden),
            "views": out,
            "seconds": round(time.time() - t0, 1)}


def main(indir, output, gate, batch, n_null, variant, mode, conds, models,
         order, horizon, probe, deep, views, coherent, untrained,
         allow_partial=False, diag=None):
    # The analysis refuses to run on a checkpoint family that did not clear the
    # validity gate.  A latent that carries nothing about where the arm went is
    # not worth probing, and a verdict read off it means nothing.  Widening to
    # PARTIAL is a declared deviation (PREREG.md G1) and is recorded below.
    allow = ("PASS", "PARTIAL") if allow_partial else ("PASS",)
    gate_report = require_pass(gate, allow)
    excluded = degenerate_tags(diag)
    rows, skipped = [], []
    for filename in sorted(glob.glob(str(Path(indir) / "arm_*_*.pt"))):
        stem = Path(filename).stem.split("_")
        if stem[1] not in models or stem[2] not in conds:
            continue
        if Path(filename).stem in excluded:
            # PREREG.md item 2: a population that never fires, or always fires,
            # cannot carry a coordinate.  Excluded and reported as excluded.
            skipped.append(Path(filename).stem)
            print("excluded (degenerate population)", Path(filename).stem,
                  flush=True)
            continue
        row = analyse(Path(filename), batch, n_null, variant, mode, order,
                      horizon, probe, deep, views, coherent, untrained)
        rows.append(row)
        for v in row["views"]:
            print(row["tag"], v["view"],
                  [(p["residual"], p["null_residual_median"], p["best_basis"],
                    p["best_kappa"]) for p in v["planes"]], flush=True)
        print(row["tag"], f"{row['seconds']}s", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(
        {"mode": mode, "variant": variant, "horizon": horizon, "probe": probe,
         "deep": deep, "coherent": bool(coherent),
         "untrained": bool(untrained), "gate": str(gate),
         "gate_verdict": gate_report.get("verdict"),
         "allow_partial": bool(allow_partial),
         "excluded_degenerate": skipped, "diagnostic": str(diag) if diag
         else None, "views": list(views), "rows": rows}, indent=2))
    print(output)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=Path, default=Path("runs/snn_v6"))
    ap.add_argument("--output", type=Path,
                    default=Path("runs/spike_torus_v6.json"))
    ap.add_argument("--gate", type=Path,
                    default=Path("runs/gate_snn_v6.json"))
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--n-null", type=int, default=4)
    ap.add_argument("--variant", default="v6")
    ap.add_argument("--mode", choices=("sequential", "deflate", "search"),
                    default="search")
    ap.add_argument("--conds", default="dark,lit,shuffle")
    ap.add_argument("--models", default="snn,rate")
    ap.add_argument("--order", default="0,1")
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--probe", type=float, default=0.4)
    ap.add_argument("--deep", type=int, default=6)
    ap.add_argument("--views", default=",".join(VIEWS))
    ap.add_argument("--coherent", action="store_true")
    ap.add_argument("--untrained", action="store_true")
    ap.add_argument("--allow-partial", action="store_true",
                    help="declared deviation: analyse a PARTIAL gate verdict")
    ap.add_argument("--diag", type=Path, default=None,
                    help="surrogate_diag report; its degenerate tags are "
                         "excluded per PREREG.md item 2")
    a = ap.parse_args()
    main(a.indir, a.output, a.gate, a.batch, a.n_null, a.variant, a.mode,
         set(a.conds.split(",")), set(a.models.split(",")),
         tuple(int(x) for x in a.order.split(",")), a.horizon, a.probe, a.deep,
         tuple(a.views.split(",")), a.coherent, a.untrained, a.allow_partial,
         a.diag)
