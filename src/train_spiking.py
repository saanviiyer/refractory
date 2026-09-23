"""Train the spiking world model and its rate twin on the two-joint arm.

Data, blackout regime, loss, optimiser, schedule, batch, sequence length and
seeds are the parent's, imported from ``vendor.train_arm_visual`` rather than
restated.  The model is the only thing that differs, which is the condition
under which "a second architecture cleared the criterion" means anything.

Two budgets are trained.  ``snn``/``rate`` match the GRU's hidden width, so
every state view has the dimension the criterion was calibrated on and the
variance-matched random-plane null is drawn from a space of the same size; they
come in about ten per cent under the GRU's parameter count.  ``snn_wide`` and
``rate_wide`` match the parameter count instead and are wider.  Neither is a
fair comparison on its own, so both are run and both are reported.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import train_arm_visual as TAV
from spiking import build_model, calibrate_threshold, n_params
from surrogate_diag import margin_report


def loss_kwargs(variant):
    """The parent's loss weighting for this variant, kept in one place."""
    return {"foreground_weight": 8 if variant != "v1" else 12,
            "distal_weight": 0 if variant in TAV.SYMMETRIC_LOSS
            else (28 if variant != "v1" else 0),
            "blackout_weight": 4 if variant in TAV.BLACKOUT_WEIGHTED else 0}


@torch.no_grad()
def evaluate(model, cond, variant, seed=870_000):
    rng = np.random.default_rng(seed + 19)
    masked, target, mask, action, _ = TAV.make_batch(
        64, 64, cond, seed, rng, variant=variant)
    pred = model(masked, mask, action)
    se = ((pred[:, :-1] - target[:, 1:]) ** 2).mean((2, 3, 4))
    weighted = TAV.prediction_loss(pred[:, :-1], target[:, 1:],
                                   mask=mask[:, :-1], **loss_kwargs(variant))
    m = mask[:, :-1]
    return {"lit_mse": float(se[m == 0].mean()),
            "dark_mse": float(se[m == 1].mean()) if (m == 1).any() else None,
            "weighted_mse": float(weighted)}


def fit(cond, seed, steps, B, T, outdir: Path, variant, model_kind, lr=2e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(model_kind)
    rng = np.random.default_rng(seed + 1_777)
    # Before the first gradient step, put the population inside the surrogate's
    # support.  Frames and actions only; no pose label is involved.
    warm = TAV.make_batch(16, 32, cond, seed, rng, variant=variant)
    calibration = calibrate_threshold(model, warm[0], warm[2], warm[3])
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    history, t0 = [], time.time()
    for step in range(1, steps + 1):
        masked, target, mask, action, _ = TAV.make_batch(
            B, T, cond, seed * 100_003 + step, rng, variant=variant)
        pred = model(masked, mask, action)
        loss = TAV.prediction_loss(pred[:, :-1], target[:, 1:],
                                   mask=mask[:, :-1], **loss_kwargs(variant))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        scheduler.step()
        if step % max(1, steps // 10) == 0:
            # Raw margins, not a summary of the surrogate: a clamped or
            # out-of-support gradient path is invisible in the loss curve and
            # in every statistic of the term itself.  Diagnostic only; the gate
            # is gate_blackout.py.
            history.append({"step": step, "mse": float(loss),
                            "margins": margin_report(model, variant, seed)})
            print(model_kind, cond, seed, step, round(float(loss), 5),
                  flush=True)
    result = {"kind": f"arm_{model_kind}", "cond": cond, "seed": seed,
              "steps": steps, "lr": lr, "batch": B, "T": T,
              "params": n_params(model), "variant": variant,
              "neurons": getattr(model, "neurons", None),
              "surrogate_width": getattr(model, "surrogate_width", None),
              "discrete_emission": bool(getattr(model, "discrete", True)),
              "threshold_calibration": calibration, "history": history,
              "eval_lit": evaluate(model, "lit", variant),
              "eval_dark": evaluate(model, "dark", variant),
              "wall_s": round(time.time() - t0, 1)}
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"arm_{model_kind}_{cond}_s{seed}"
    torch.save(model.state_dict(), outdir / (tag + ".pt"))
    (outdir / (tag + ".json")).write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", default="dark")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--outdir", type=Path, default=Path("runs/snn_v6"))
    ap.add_argument("--variant", choices=tuple(TAV.ENVIRONMENTS), default="v6")
    ap.add_argument("--model", default="snn")
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()
    for cond in args.conds.split(","):
        for seed in map(int, args.seeds.split(",")):
            tag = f"arm_{args.model}_{cond}_s{seed}"
            if (args.outdir / (tag + ".json")).exists():
                print("skip", tag)
                continue
            fit(cond, seed, args.steps, args.batch, args.T, args.outdir,
                args.variant, args.model, args.lr)
