"""Score amendment A4 against the confirmatory run.

Written before the fresh seeds finished training, so the verdict is produced by
code rather than by reading a table and deciding what it shows.  A4 fixes three
outcomes and this file can only emit those three: the criteria are transcribed
from the amendment, and a result that satisfies none of them comes out
"uninformative" rather than being argued into one of the other two.

The primary prediction is deliberately narrow.  It names one joint and one pair
of views, because that is where the discovery observation sat; a different pair
disagreeing elsewhere is a new observation and is reported separately rather
than counted as a confirmation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

A3_FLOOR = 0.60          # PREREG A3, itself the band H1 already fixes
PRIMARY_JOINT = 1        # A4: where the discovery observation sat
PRIMARY_PAIR = ("s", "i")
MAJORITY = 6             # of 8 seeds
P1_MAJORITY = 7          # of 8 seeds, for the secondary prediction


def cell_summary(report, kind):
    """Per (view, joint): majority family, its count, and the A3 floor test."""
    rows = [r for r in report["rows"]
            if r.get("cond") == "dark" and r.get("model_kind") == kind]
    out = {}
    for view in report["views"]:
        for joint in (1, 2):
            got = [f for r in rows for f in r["fits"]
                   if f["view"] == view and f["joint"] == joint]
            if not got:
                continue
            n_rot = sum(g["family"] == "rotation" for g in got)
            win = [min(g["rotation"]["residual"], g["translation"]["residual"])
                   for g in got]
            median_win = float(np.median(win))
            out[(view, joint)] = {
                "n": len(got), "n_rotation": n_rot,
                "n_translation": len(got) - n_rot,
                "majority": "rotation" if n_rot * 2 > len(got) else
                            ("translation" if (len(got) - n_rot) * 2 > len(got)
                             else "tied"),
                "majority_count": max(n_rot, len(got) - n_rot),
                "median_win_residual": round(median_win, 4),
                "clears_a3_floor": bool(median_win < A3_FLOOR),
                "median_margin": round(float(np.median(
                    [g["margin"] for g in got])), 4)}
    return out


def score_primary(summary):
    """A4's primary prediction. Returns one of confirmed/refuted/uninformative."""
    a, b = PRIMARY_PAIR
    ka, kb = (a, PRIMARY_JOINT), (b, PRIMARY_JOINT)
    if ka not in summary or kb not in summary:
        return {"verdict": "uninformative",
                "reason": f"view {a} or {b} absent for joint {PRIMARY_JOINT}"}
    sa, sb = summary[ka], summary[kb]
    if not (sa["clears_a3_floor"] and sb["clears_a3_floor"]):
        return {"verdict": "uninformative",
                "reason": "fewer than both views clear the A3 floor",
                f"{a}_residual": sa["median_win_residual"],
                f"{b}_residual": sb["median_win_residual"]}
    ok = (sa["majority"] == "translation"
          and sa["majority_count"] >= MAJORITY
          and sb["majority"] == "rotation"
          and sb["majority_count"] >= MAJORITY)
    return {"verdict": "confirmed" if ok else "refuted",
            "reason": ("opposite majorities at the pre-registered strength"
                       if ok else
                       "both views clear the floor but the predicted opposite "
                       "majorities did not appear"),
            f"{a}_majority": f"{sa['majority']} {sa['majority_count']}/{sa['n']}",
            f"{b}_majority": f"{sb['majority']} {sb['majority_count']}/{sb['n']}"}


def score_secondary(summary):
    p1 = {}
    for joint in (1, 2):
        k = ("s", joint)
        if k in summary:
            p1[f"joint{joint}"] = {
                "majority": summary[k]["majority"],
                "count": summary[k]["majority_count"],
                "holds": (summary[k]["majority"] == "translation"
                          and summary[k]["majority_count"] >= P1_MAJORITY)}
    p2 = {}
    for joint in (1, 2):
        k = ("r", joint)
        if k in summary:
            p2[f"joint{joint}"] = {
                "median_win_residual": summary[k]["median_win_residual"],
                "clears_a3_floor": summary[k]["clears_a3_floor"],
                # P2 predicts r does NOT clear the floor.
                "holds": not summary[k]["clears_a3_floor"]}
    return {"P1_s_view_returns_translation": p1,
            "P2_primary_view_finds_nothing": p2}


def other_disagreements(summary):
    """Floor-clearing view pairs that disagree, outside the pre-registered one.

    Recorded so a disagreement elsewhere is visible as the new observation it
    would be, rather than being quietly counted as a confirmation.
    """
    clearing = {k: v for k, v in summary.items()
                if v["clears_a3_floor"] and v["majority"] != "tied"
                and k[0] != "all"}
    out = []
    keys = sorted(clearing)
    for i, ka in enumerate(keys):
        for kb in keys[i + 1:]:
            if ka[1] != kb[1]:
                continue
            if clearing[ka]["majority"] != clearing[kb]["majority"]:
                pair = tuple(sorted((ka[0], kb[0])))
                out.append({
                    "joint": ka[1], "views": list(pair),
                    "is_the_preregistered_pair": (
                        pair == tuple(sorted(PRIMARY_PAIR))
                        and ka[1] == PRIMARY_JOINT),
                    f"{ka[0]}": f"{clearing[ka]['majority']} "
                                f"{clearing[ka]['majority_count']}/{clearing[ka]['n']}",
                    f"{kb[0]}": f"{clearing[kb]['majority']} "
                                f"{clearing[kb]['majority_count']}/{clearing[kb]['n']}"})
    return out


def score_a5(summary):
    """A5's own prediction, which is not A4's.

    A4 asks whether two views disagree.  A5 asks something narrower: does the
    `s` view's translation verdict already exist, and fit, before any learning?
    Reusing A4's scorer on untrained data answers "does the untrained model
    reproduce the A4 pattern", which is a different question and not the one
    A5 pre-registered.
    """
    k = ("s", PRIMARY_JOINT)
    if k not in summary:
        return {"verdict": "uninformative", "reason": "s view absent"}
    d = summary[k]
    geometric = (d["majority"] == "translation"
                 and d["majority_count"] >= MAJORITY
                 and d["clears_a3_floor"])
    return {"verdict": "geometric" if geometric else "learned",
            "reason": ("the untrained s view already returns translation at "
                       "the pre-registered strength and fits"
                       if geometric else
                       "the untrained s view does not both hold the majority "
                       "and clear the A3 floor, so geometry alone does not "
                       "produce the trained verdict"),
            "s_majority": f"{d['majority']} {d['majority_count']}/{d['n']}",
            "s_median_win_residual": d["median_win_residual"],
            "s_clears_a3_floor": d["clears_a3_floor"]}


def score_floor_discrimination(summary):
    """Does the A3 floor separate an untrained model from a trained one?

    Recorded because the floor was introduced (A3) as a quality check on the
    fit and is easy to read as evidence that the criterion found something
    real.  If a random network clears it, it is not that.
    """
    clearing = sorted(f"{v}_j{j}" for (v, j), d in summary.items()
                      if d["clears_a3_floor"])
    return {"views_clearing_floor_untrained": clearing,
            "n_clearing": len(clearing),
            "floor_separates_trained_from_untrained": len(clearing) == 0}


def main(family_paths, gates, output):
    cells = []
    for kind, fam_path in family_paths:
        path = Path(fam_path)
        if not path.exists():
            cells.append({"kind": kind, "status": "not analysed",
                          "reason": f"{fam_path} absent (gate refusal?)"})
            continue
        report = json.loads(path.read_text())
        summary = cell_summary(report, kind)
        gate = json.loads(Path(gates[kind]).read_text())
        cells.append({
            "kind": kind,
            "gate_verdict": gate["verdict"],
            "gate_n_pass": gate["n_beating_persistence"],
            "gate_median_ratio": gate["median_ratio"],
            "analysed_under_allow_partial": bool(report.get("allow_partial")),
            "n_seeds": len({r["seed"] for r in report["rows"]}),
            "primary": score_primary(summary),
            "a5_untrained": score_a5(summary),
            "a5_floor_discrimination": score_floor_discrimination(summary),
            "secondary": score_secondary(summary),
            "other_floor_clearing_disagreements": other_disagreements(summary),
            "per_view": {f"{v}_j{j}": d for (v, j), d in sorted(summary.items())}})
    verdicts = [c["primary"]["verdict"] for c in cells if "primary" in c]
    overall = ("confirmed" if "confirmed" in verdicts else
               "refuted" if "refuted" in verdicts else "uninformative")
    out = {"amendment": "A4",
           "prediction": (f"on joint {PRIMARY_JOINT}, view {PRIMARY_PAIR[0]} "
                          f"returns translation >= {MAJORITY}/8 and view "
                          f"{PRIMARY_PAIR[1]} returns rotation >= {MAJORITY}/8, "
                          "both clearing the A3 floor"),
           "a3_floor": A3_FLOOR, "overall": overall, "cells": cells}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(out, indent=2))
    print(f"A4: {overall}")
    for c in cells:
        if "primary" in c:
            print(f"  {c['kind']:6s} gate {c['gate_verdict']:8s} "
                  f"-> {c['primary']['verdict']}: {c['primary']['reason']}")
    print(output)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--snn-family", default="runs/spike_family_conf_v7_snn.json")
    ap.add_argument("--rate-family",
                    default="runs/spike_family_conf_v7_rate.json")
    ap.add_argument("--snn-gate", default="runs/gate_snn_conf_v7.json")
    ap.add_argument("--rate-gate", default="runs/gate_rate_conf_v7.json")
    ap.add_argument("--output", default="runs/a4_verdict.json")
    a = ap.parse_args()
    main([("snn", a.snn_family), ("rate", a.rate_family)],
         {"snn": a.snn_gate, "rate": a.rate_gate}, a.output)
