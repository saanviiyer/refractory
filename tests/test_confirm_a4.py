"""The A4 scorer can only emit A4's three outcomes, and cannot be argued into one.

Written against synthetic reports before the confirmatory seeds finished, so the
scoring rule is fixed independently of what the data turned out to be.
"""
from __future__ import annotations

import confirm_a4 as C


def fits(view, joint, n_rot, n_tra, rot_res, tra_res):
    out = []
    for i in range(n_rot + n_tra):
        rot = rot_res if i < n_rot else rot_res + 0.4
        tra = tra_res + 0.4 if i < n_rot else tra_res
        out.append({"view": view, "joint": joint,
                    "rotation": {"residual": rot},
                    "translation": {"residual": tra},
                    "family": "rotation" if i < n_rot else "translation",
                    "margin": abs(rot - tra)})
    return out


def report(spec, views=("s", "i", "r")):
    rows = [{"cond": "dark", "model_kind": "snn", "seed": 0, "fits": []}]
    for (view, joint), (n_rot, n_tra, rr, tr) in spec.items():
        rows[0]["fits"] += fits(view, joint, n_rot, n_tra, rr, tr)
    return {"views": list(views), "rows": rows, "allow_partial": False}


def test_the_predicted_pattern_confirms():
    r = report({("s", 1): (0, 8, 0.9, 0.25), ("i", 1): (7, 1, 0.30, 0.9)})
    assert C.score_primary(C.cell_summary(r, "snn"))["verdict"] == "confirmed"


def test_agreement_between_floor_clearing_views_refutes():
    r = report({("s", 1): (0, 8, 0.9, 0.25), ("i", 1): (0, 8, 0.9, 0.30)})
    assert C.score_primary(C.cell_summary(r, "snn"))["verdict"] == "refuted"


def test_a_weak_majority_refutes_rather_than_confirming():
    # 5/8 is below the pre-registered 6/8, and must not be argued into a pass.
    r = report({("s", 1): (3, 5, 0.9, 0.25), ("i", 1): (7, 1, 0.30, 0.9)})
    assert C.score_primary(C.cell_summary(r, "snn"))["verdict"] == "refuted"


def test_a_view_that_fits_nothing_is_uninformative_not_refuted():
    # The discovery run's failure mode: opposite verdicts, but above the floor.
    r = report({("s", 1): (0, 8, 0.99, 0.95), ("i", 1): (7, 1, 0.93, 0.99)})
    v = C.score_primary(C.cell_summary(r, "snn"))
    assert v["verdict"] == "uninformative"
    assert "floor" in v["reason"]


def test_disagreement_on_the_other_joint_does_not_confirm():
    r = report({("s", 1): (0, 8, 0.9, 0.25), ("i", 1): (0, 8, 0.9, 0.30),
                ("s", 2): (0, 8, 0.9, 0.25), ("i", 2): (7, 1, 0.30, 0.9)})
    summary = C.cell_summary(r, "snn")
    assert C.score_primary(summary)["verdict"] == "refuted"
    others = C.other_disagreements(summary)
    assert any(o["joint"] == 2 and not o["is_the_preregistered_pair"]
               for o in others)


def test_P2_holds_only_when_the_primary_view_finds_nothing():
    r = report({("r", 1): (5, 3, 0.95, 0.99)})
    assert C.score_secondary(C.cell_summary(r, "snn"))[
        "P2_primary_view_finds_nothing"]["joint1"]["holds"] is True
    r = report({("r", 1): (5, 3, 0.20, 0.99)})
    assert C.score_secondary(C.cell_summary(r, "snn"))[
        "P2_primary_view_finds_nothing"]["joint1"]["holds"] is False
