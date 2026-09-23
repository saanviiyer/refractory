"""Every number that appears in prose is generated here, from run JSON.

The parent shipped exactly one number that was typed rather than generated, and
that was the one that turned out to be wrong: a horizon trend read off a stale
environment and a single seed, which sat in a draft for days because nothing
regenerated it.  The rule that follows is not "check numbers more carefully".
It is that an un-macroed number is a defect by construction, so this file is
the only place a number is allowed to come from.

A macro whose source run is missing is omitted rather than defaulted.  A draft
that quotes it then fails to build, which is the intended behaviour: a claim
whose run has not been done should disappear, not degrade to a plausible value.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MISSING = object()


def load(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _dark(rows, kind=None):
    out = [r for r in rows if r.get("cond") == "dark"]
    return [r for r in out if kind is None or r.get("model_kind") == kind]


def gate_macros(report, prefix):
    if report is None:
        return {}
    return {f"{prefix}Verdict": report["verdict"],
            f"{prefix}NPass": report["n_beating_persistence"],
            f"{prefix}NModels": report["n_models"],
            f"{prefix}MedianRatio": report["median_ratio"]}


def torus_macros(report, prefix, kind):
    """Per-view medians of the criterion, across seeds, for one model kind."""
    if report is None:
        return {}
    rows = _dark(report["rows"], kind)
    if not rows:
        return {}
    out = {f"{prefix}NSeeds": len(rows)}
    for view in report["views"]:
        for joint in (0, 1):
            got = [next(v for v in r["views"] if v["view"] == view)
                   ["planes"][joint] for r in rows]
            tag = f"{prefix}{view.capitalize()}P{joint + 1}"
            out[f"{tag}Residual"] = round(
                float(np.median([g["residual"] for g in got])), 4)
            out[f"{tag}NullResidual"] = round(
                float(np.median([g["null_residual_median"] for g in got])), 4)
            out[f"{tag}Kappa"] = round(
                float(np.median([g["best_kappa"] for g in got])), 4)
            out[f"{tag}NBeatNull"] = int(sum(g["beats_null"] for g in got))
            bases = {tuple(g["best_basis"]) for g in got}
            out[f"{tag}BasisUnanimous"] = len(bases) == 1
            out[f"{tag}Basis"] = (list(bases.pop()) if len(bases) == 1
                                  else None)
    return out


def family_macros(report, prefix, kind):
    if report is None:
        return {}
    rows = _dark(report["rows"], kind)
    if not rows:
        return {}
    out = {f"{prefix}NSeeds": len(rows)}
    for view in report["views"]:
        for joint in (1, 2):
            got = [f for r in rows for f in r["fits"]
                   if f["view"] == view and f["joint"] == joint]
            if not got:
                continue
            tag = f"{prefix}{view.capitalize()}J{joint}"
            out[f"{tag}NRotation"] = sum(g["family"] == "rotation"
                                         for g in got)
            out[f"{tag}N"] = len(got)
            out[f"{tag}RotResidual"] = round(float(np.median(
                [g["rotation"]["residual"] for g in got])), 4)
            out[f"{tag}TraResidual"] = round(float(np.median(
                [g["translation"]["residual"] for g in got])), 4)
            # A3: the winning family's own residual.  A verdict whose winner
            # sits near the no-transform reference of 1.0 is a coin flip
            # between two families that both explain nothing.
            winners = [min(g["rotation"]["residual"],
                           g["translation"]["residual"]) for g in got]
            out[f"{tag}WinResidual"] = round(float(np.median(winners)), 4)
            out[f"{tag}ClearsA3Floor"] = bool(np.median(winners) < 0.60)
            out[f"{tag}Margin"] = round(float(np.median(
                [g["margin"] for g in got])), 4)
    return out


def family_agreement_macros(report, prefix, kind, reference="r"):
    """Per-seed family agreement between each view and the reference view.

    A3: this is only readable when both views clear the residual floor.  A view
    that fits nothing agrees or disagrees at random, so the count is recorded
    alongside whether each side of the comparison actually fit.
    """
    if report is None:
        return {}
    rows = _dark(report["rows"], kind)
    if not rows:
        return {}
    out = {}
    for view in report["views"]:
        if view == reference:
            continue
        agree = total = 0
        for r in rows:
            ref = {(f["joint"]): f for f in r["fits"] if f["view"] == reference}
            oth = {(f["joint"]): f for f in r["fits"] if f["view"] == view}
            for j in set(ref) & set(oth):
                total += 1
                agree += ref[j]["family"] == oth[j]["family"]
        out[f"{prefix}FamAgree{view.capitalize()}"] = agree
        out[f"{prefix}FamAgreeTotal{view.capitalize()}"] = total
    return out


def agreement_macros(report, prefix, kind, reference="r"):
    """How often does a view's verdict match the primary view's?

    This is the project's own claim in one number per view: if the criterion
    reads a property of the model, every view agrees with the state the decoder
    reads; if it reads a property of the analyst's choice, they do not.
    """
    if report is None:
        return {}
    rows = _dark(report["rows"], kind)
    if not rows or reference not in report["views"]:
        return {}
    out = {}
    for view in report["views"]:
        if view == reference:
            continue
        agree = total = 0
        for r in rows:
            ref = next(v for v in r["views"] if v["view"] == reference)
            other = next(v for v in r["views"] if v["view"] == view)
            for a, b in zip(ref["planes"], other["planes"]):
                total += 1
                agree += (a["best_basis"] == b["best_basis"]
                          and a["beats_null"] == b["beats_null"])
        out[f"{prefix}Agree{view.capitalize()}"] = agree
        out[f"{prefix}AgreeTotal{view.capitalize()}"] = total
    return out


def diag_macros(report, prefix, kind):
    """Population health, and how graded the control's emission actually is."""
    if report is None:
        return {}
    rows = [r for r in report["rows"] if r["model_kind"] == kind]
    if not rows:
        return {}
    return {f"{prefix}NDegenerate": sum(r["degenerate"] for r in rows),
            f"{prefix}N": len(rows),
            f"{prefix}FiringRate": round(float(np.median(
                [r["mean_firing_rate"] for r in rows])), 4),
            f"{prefix}MarginP50": round(float(np.median(
                [r["abs_margin_p50"] for r in rows])), 4),
            f"{prefix}GradedMedian": round(float(np.median(
                [r["frac_emission_graded"] for r in rows])), 4),
            f"{prefix}GradedMin": round(float(np.min(
                [r["frac_emission_graded"] for r in rows])), 4),
            f"{prefix}GradedMax": round(float(np.max(
                [r["frac_emission_graded"] for r in rows])), 4)}


def training_macros(candidates, prefix, kind):
    """Training summary for one kind, from whichever run directory holds it.

    ``run_pilot.sh`` writes a model and its twin into one directory named after
    the model, so the twin does not live under a path named for itself.
    """
    got = []
    for indir in ([candidates] if isinstance(candidates, (str, Path))
                  else candidates):
        got = [json.loads(p.read_text())
               for p in sorted(Path(indir).glob(f"arm_{kind}_dark_s*.json"))]
        if got:
            break
    if not got:
        return {}
    return {f"{prefix}Params": got[0]["params"],
            f"{prefix}Neurons": got[0]["neurons"],
            f"{prefix}Steps": got[0]["steps"],
            f"{prefix}NSeeds": len(got),
            f"{prefix}DarkMse": round(float(np.median(
                [g["eval_dark"]["dark_mse"] for g in got])), 5)}


def curve_macros(candidates, prefix, kind):
    """Loss-curve endpoints and the change over the final quarter of training.

    "Is it still improving at the budget?" is the first question asked of any
    negative result, so the answer is generated rather than eyeballed off a
    plot.
    """
    got = []
    for indir in candidates:
        got = [json.loads(p.read_text())
               for p in sorted(Path(indir).glob(f"arm_{kind}_dark_s*.json"))]
        if got:
            break
    if not got:
        return {}
    firsts = [d["history"][0]["mse"] for d in got]
    lasts = [d["history"][-1]["mse"] for d in got]
    # Margins are logged through training, so "did the model train itself out
    # of its own emission regime?" is answerable rather than inferred.
    m_first = [d["history"][0]["margins"]["abs_margin_p50"] for d in got
               if "margins" in d["history"][0]]
    m_last = [d["history"][-1]["margins"]["abs_margin_p50"] for d in got
              if "margins" in d["history"][-1]]
    # Percentage fall over the last four logged points; negative means it rose.
    drops = [(d["history"][-4]["mse"] - d["history"][-1]["mse"])
             / d["history"][-4]["mse"] * 100 for d in got
             if len(d["history"]) >= 4]
    out = {}
    if m_first and m_last:
        # How many seeds end outside the band a 0.25-width sigmoid keeps a
        # graded emission in?  ln(0.98/0.02)/0.25 = 15.6.
        out[f"{prefix}NSeedsAboveGradedBand"] = int(
            sum(1 for m in m_last if m > 15.6))
        out[f"{prefix}NSeeds"] = len(m_last)
        out[f"{prefix}MarginFirst"] = round(float(np.median(m_first)), 2)
        out[f"{prefix}MarginLast"] = round(float(np.median(m_last)), 2)
        out[f"{prefix}MarginGrowth"] = round(
            float(np.median(m_last) / max(np.median(m_first), 1e-9)), 1)
    return {**out,
            f"{prefix}MseFirst": round(float(np.median(firsts)), 4),
            f"{prefix}MseLastMin": round(float(np.min(lasts)), 4),
            f"{prefix}MseLastMax": round(float(np.max(lasts)), 4),
            f"{prefix}FinalQuarterPctMedian": round(float(np.median(drops)), 1),
            f"{prefix}FinalQuarterPctMin": round(float(np.min(drops)), 1),
            f"{prefix}FinalQuarterPctMax": round(float(np.max(drops)), 1)}


def parent_family_macros(parent_runs, variant="v7"):
    """The parent's own family verdicts, read from its run JSON.

    INSTRUMENT.md quotes these; quoting the document would be transcription, so
    they are regenerated from the run the document was written from.
    """
    import numpy as _np
    path = Path(parent_runs) / f"arm_family_{variant}.json"
    if not path.exists():
        return {}
    rows = [r for r in json.loads(path.read_text())["rows"]
            if r.get("cond") == "dark" and r.get("model_kind") == "gru"]
    if not rows:
        return {}
    out = {f"parentFam{variant.upper()}NSeeds": len(rows)}
    for j in (1, 2):
        got = [x for r in rows for x in r["joints"] if x["joint"] == j]
        if not got:
            continue
        win = [min(g["rotation"]["residual"], g["translation"]["residual"])
               for g in got]
        out[f"parentFam{variant.upper()}J{j}WinResidual"] = round(
            float(_np.median(win)), 4)
        out[f"parentFam{variant.upper()}J{j}NTranslation"] = sum(
            g["family"] == "translation" for g in got)
    return out


def parent_baseline_macros(parent_runs, variant="v6", kind="gru"):
    """The parent's own training numbers, for the comparison in prose."""
    got = [json.loads(p.read_text()) for p in
           sorted(Path(parent_runs).glob(f"arm_{variant}/arm_{kind}_dark_s*.json"))]
    if not got:
        return {}
    drops = [(d["history"][-4]["mse"] - d["history"][-1]["mse"])
             / d["history"][-4]["mse"] * 100 for d in got
             if len(d["history"]) >= 4]
    tag = f"parentGru{variant.upper()}"
    return {f"{tag}DarkMse": round(float(np.median(
                [d["eval_dark"]["dark_mse"] for d in got])), 5),
            f"{tag}NSeeds": len(got),
            f"{tag}FinalQuarterPctMedian": round(float(np.median(drops)), 1),
            f"{tag}FinalQuarterPctMin": round(float(np.min(drops)), 1),
            f"{tag}FinalQuarterPctMax": round(float(np.max(drops)), 1)}


def a4_macros(report):
    """The confirmatory verdict, as the scorer emitted it."""
    if report is None:
        return {}
    out = {"a4Overall": report["overall"], "a4Floor": report["a3_floor"]}
    for c in report["cells"]:
        k = c["kind"].capitalize()
        if "primary" not in c:
            continue
        out[f"a4{k}Gate"] = c["gate_verdict"]
        out[f"a4{k}GateNPass"] = c["gate_n_pass"]
        out[f"a4{k}GateRatio"] = c["gate_median_ratio"]
        out[f"a4{k}Primary"] = c["primary"]["verdict"]
        for side in ("s", "i"):
            if f"{side}_majority" in c["primary"]:
                out[f"a4{k}{side.upper()}Majority"] = c["primary"][f"{side}_majority"]
            if f"{side}_residual" in c["primary"]:
                out[f"a4{k}{side.upper()}Residual"] = c["primary"][f"{side}_residual"]
        p1 = c["secondary"]["P1_s_view_returns_translation"]
        p2 = c["secondary"]["P2_primary_view_finds_nothing"]
        out[f"a4{k}P1Holds"] = all(v["holds"] for v in p1.values())
        out[f"a4{k}P2Holds"] = all(v["holds"] for v in p2.values())
        for j, v in p2.items():
            out[f"a4{k}RResidual{j.capitalize()}"] = v["median_win_residual"]
        out[f"a4{k}NOtherDisagreements"] = len(
            [o for o in c["other_floor_clearing_disagreements"]
             if not o["is_the_preregistered_pair"]])
        # Trained per-view residuals, so the trained/untrained comparison in
        # prose is generated on both sides rather than half-typed.
        for key, d in c["per_view"].items():
            tag = key.replace("_", "").upper()
            out[f"a4{k}Trained{tag}Residual"] = d["median_win_residual"]
            out[f"a4{k}Trained{tag}Majority"] = (
                f"{d['majority']} {d['majority_count']}/{d['n']}")
    return out


def a5_macros(report):
    """The untrained control, and what it revealed about the A3 floor."""
    if report is None:
        return {}
    out = {}
    for c in report["cells"]:
        if "a5_untrained" not in c:
            continue
        k = c["kind"].capitalize()
        a, f = c["a5_untrained"], c["a5_floor_discrimination"]
        out[f"a5{k}Verdict"] = a["verdict"]
        out[f"a5{k}SMajority"] = a["s_majority"]
        out[f"a5{k}SResidual"] = a["s_median_win_residual"]
        out[f"a5{k}SClearsFloor"] = a["s_clears_a3_floor"]
        out[f"a5{k}NUntrainedClearingFloor"] = f["n_clearing"]
        out[f"a5{k}FloorSeparates"] = f["floor_separates_trained_from_untrained"]
        for key, d in c["per_view"].items():
            out[f"a5{k}Untrained{key.replace('_','').upper()}Residual"] = \
                d["median_win_residual"]
            out[f"a5{k}Untrained{key.replace('_','').upper()}Majority"] = \
                f"{d['majority']} {d['majority_count']}/{d['n']}"
    return out


def build(runs: Path, parent_runs=None):
    """Macros for one runs directory.

    ``parent_runs`` is opt-in: the parent's baseline is a different repository's
    result, and folding it in unconditionally would break the invariant this
    module exists for --- that a macro whose run is missing is absent, not
    defaulted.
    """
    macros = {}
    macros.update(a4_macros(load(runs / "a4_verdict.json")))
    macros.update(a5_macros(load(runs / "a5_untrained_verdict.json")))
    cal = load(runs / "gate_calibration.json")
    if cal is not None:
        macros["gateCalibrated"] = cal["calibrated"]
        macros["gateCalAgreeing"] = cal["n_agreeing"]
        macros["gateCalChecked"] = cal["n_checked"]
        for row in cal["rows"]:
            if row.get("got") is None:
                continue
            name = Path(row["indir"]).name.replace("_", "").capitalize()
            macros[f"parent{name}Verdict"] = row["got"]
            macros[f"parent{name}Ratio"] = row["median_ratio"]
            macros[f"parent{name}NPass"] = row["n_beating_persistence"]
    if parent_runs is not None:
        macros.update(parent_baseline_macros(parent_runs, "v6"))
        macros.update(parent_family_macros(parent_runs, "v7"))
    for variant in ("v6", "v7"):
        macros.update(gate_macros(load(runs / f"gate_snn_{variant}.json"),
                                  f"gateSnn{variant.upper()}"))
        macros.update(gate_macros(load(runs / f"gate_rate_{variant}.json"),
                                  f"gateRate{variant.upper()}"))
        macros.update(gate_macros(load(runs / f"gate_ratesoft_{variant}.json"),
                                  f"gateSoft{variant.upper()}"))
        for kind, label in (("snn", "Snn"), ("rate", "Rate"),
                            ("ratesoft", "Soft")):
            torus = (load(runs / f"spike_torus_{variant}_{kind}.json")
                     or load(runs / f"spike_torus_{variant}.json"))
            macros.update(torus_macros(torus, f"torus{label}{variant.upper()}",
                                       kind))
            macros.update(agreement_macros(
                torus, f"torus{label}{variant.upper()}", kind))
            macros.update(training_macros(
                [runs / f"{kind}_{variant}", runs / f"snn_{variant}",
                 runs / f"snn_wide_{variant}"],
                f"train{label}{variant.upper()}", kind))
            macros.update(curve_macros(
                [runs / f"{kind}_{variant}", runs / f"snn_{variant}",
                 runs / f"snn_wide_{variant}"],
                f"train{label}{variant.upper()}", kind))
        diag = load(runs / f"surrogate_diag_{variant}.json")
        for kind, label in (("snn", "Snn"), ("rate", "Rate"),
                            ("ratesoft", "Soft")):
            macros.update(diag_macros(diag, f"diag{label}{variant.upper()}",
                                      kind))
        # Analyses are written per cell, since PREREG G1 gates per cell.  The
        # combined filename is still read so earlier runs remain loadable.
        for kind, label in (("snn", "Snn"), ("rate", "Rate"),
                            ("ratesoft", "Soft")):
            fam = (load(runs / f"spike_family_{variant}_{kind}.json")
                   or load(runs / f"spike_family_{variant}.json"))
            macros.update(family_macros(fam, f"fam{label}{variant.upper()}",
                                        kind))
            macros.update(family_agreement_macros(
                fam, f"fam{label}{variant.upper()}", kind))
            if fam is not None:
                macros[f"fam{label}{variant.upper()}GateVerdict"] = \
                    fam.get("gate_verdict")
                macros[f"fam{label}{variant.upper()}AllowPartial"] = \
                    fam.get("allow_partial")
    # How much easier is the bounded arm for each architecture?  Derived from
    # macros already generated, so the comparison in prose is not typed.
    pairs = (("gateSnnV6MedianRatio", "gateSnnV7MedianRatio", "improveSnn"),
             ("gateRateV6MedianRatio", "gateRateV7MedianRatio", "improveRate"),
             ("parentArmv6Ratio", "parentArmv7Ratio", "improveGru"))
    for v6key, v7key, name in pairs:
        a, b = macros.get(v6key), macros.get(v7key)
        if a and b:
            macros[name] = round(float(a) / float(b), 2)
    return macros


def main(runs, output, parent_runs=None):
    macros = build(Path(runs), parent_runs)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(macros, indent=2, sort_keys=True))
    print(f"{len(macros)} macros -> {output}")
    return macros


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--output", default="paper/numbers.json")
    ap.add_argument("--parent-runs", default="../used_coordinates/runs")
    a = ap.parse_args()
    main(a.runs, a.output, a.parent_runs)
