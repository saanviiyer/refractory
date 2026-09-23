"""A macro whose run is missing must disappear, not degrade to a default."""
from __future__ import annotations

import json

import make_numbers as mn


def test_no_runs_yields_no_macros(tmp_path):
    assert mn.build(tmp_path) == {}


def test_a_partial_run_directory_yields_only_what_it_supports(tmp_path):
    (tmp_path / "gate_snn_v6.json").write_text(json.dumps(
        {"verdict": "PASS", "n_models": 8, "n_beating_persistence": 8,
         "median_ratio": 0.61}))
    macros = mn.build(tmp_path)
    assert macros["gateSnnV6Verdict"] == "PASS"
    assert macros["gateSnnV6MedianRatio"] == 0.61
    # Nothing downstream of a run that has not happened.
    assert not [k for k in macros if k.startswith("torus")]
    assert not [k for k in macros if k.startswith("fam")]


def _torus_row(seed, kind, bases, residuals):
    return {"cond": "dark", "model_kind": kind, "seed": seed,
            "views": [{"view": v,
                       "planes": [{"residual": residuals[v][j],
                                   "null_residual_median": 0.99,
                                   "best_kappa": 0.95,
                                   "beats_null": residuals[v][j] < 0.99,
                                   "best_basis": bases[v][j]}
                                  for j in (0, 1)]}
                      for v in ("r", "v")]}


def test_agreement_counts_views_against_the_primary(tmp_path):
    rows = [
        # Seed 0: the two views agree on both planes.
        _torus_row(0, "snn", {"r": [[1, 0], [0, 1]], "v": [[1, 0], [0, 1]]},
                   {"r": [0.2, 0.25], "v": [0.3, 0.35]}),
        # Seed 1: the membrane view finds a different basis for plane 2.
        _torus_row(1, "snn", {"r": [[1, 0], [0, 1]], "v": [[1, 0], [1, 1]]},
                   {"r": [0.2, 0.25], "v": [0.3, 0.35]}),
    ]
    (tmp_path / "spike_torus_v6.json").write_text(json.dumps(
        {"views": ["r", "v"], "rows": rows}))
    macros = mn.build(tmp_path)
    assert macros["torusSnnV6AgreeTotalV"] == 4
    assert macros["torusSnnV6AgreeV"] == 3
    assert macros["torusSnnV6RP1Residual"] == 0.2
    assert macros["torusSnnV6VP2BasisUnanimous"] is False


def test_unanimity_is_reported_not_assumed(tmp_path):
    rows = [_torus_row(s, "snn", {"r": [[1, 0], [0, 1]],
                                  "v": [[1, 0], [0, 1]]},
                       {"r": [0.2, 0.25], "v": [0.3, 0.35]}) for s in (0, 1)]
    (tmp_path / "spike_torus_v6.json").write_text(json.dumps(
        {"views": ["r", "v"], "rows": rows}))
    macros = mn.build(tmp_path)
    assert macros["torusSnnV6RP1BasisUnanimous"] is True
    assert macros["torusSnnV6RP1Basis"] == [1, 0]
    assert macros["torusSnnV6NSeeds"] == 2


# --- the project's own rule, enforced ---------------------------------------
# The parent shipped exactly one number that was typed rather than generated and
# it was the one that turned out to be wrong.  This test is that rule made
# mechanical: a figure in FINDINGS.md that no macro backs is a defect.  It has
# already earned its keep once, catching a convergence range read off three
# seeds when eight were available.

# Structural values: dates, seed counts, hyperparameters stated as
# configuration, PREREG band edges, and figures defined in the prose itself.
STRUCTURAL = {
    "31", "2026", "0", "1", "2", "4", "8", "16", "32", "64", "128", "1600",
    "0.02", "0.98", "0.5", "1.0", "2.25", "0.13", "1.9", "0.9995", "0.9892",
    # Configuration and PREREG band edges, not measurements:
    "0.25",   # rate_soft emission width, fixed in amendment A1
    "0.70",   # A1's validity threshold on the graded fraction
    "15.6",   # ln(0.98/0.02)/0.25, where that emission leaves the graded band
}


def _figures(text):
    import re
    return {m.group(1) for m in
            re.finditer(r"(?<![\w.])(-?\d+\.\d+)(?![\w])", text)}


def test_every_figure_in_findings_is_macro_backed():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    numbers = root / "paper" / "numbers.json"
    findings = root / "FINDINGS.md"
    if not numbers.exists() or not findings.exists():
        import pytest
        pytest.skip("no generated numbers yet")
    macros = json.loads(numbers.read_text())
    backed = {str(v) for v in macros.values()}
    backed |= {f"{v:g}" for v in macros.values() if isinstance(v, float)}
    unbacked = sorted(f for f in _figures(findings.read_text())
                      if f not in backed and f not in STRUCTURAL)
    assert not unbacked, (
        f"figures in FINDINGS.md with no macro in paper/numbers.json: "
        f"{unbacked}. Generate them in src/make_numbers.py rather than "
        f"typing them.")
