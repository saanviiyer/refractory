#!/bin/bash
# PREREG A1: the graded control on v6.
#
# The first rate twin emits a graded value on only about a quarter of steps,
# because sigmoid(4 * margin) saturates at the margins training produces, so it
# shares the discreteness it was built to remove.  This model emits
# sigmoid(0.25 * margin) and is differentiated exactly, removing the discrete
# forward, the forward/backward mismatch and the weak gradient together.
#
# It therefore separates spiking machinery from architecture, not discreteness
# from everything else.  A1 states that, and states the validity condition:
# below a median graded fraction of 0.70 this is just another saturated model.
set -u
cd "$(dirname "$0")"
export PYTHONPATH=src:src/vendor
export OMP_NUM_THREADS=1
OUT=runs/snn_v6
mkdir -p "$OUT" logs

echo "=== A1: train the graded control, 8 seeds, v6, 1600 steps ==="
for s in 0 1 2 3 4 5 6 7; do
  ( python3 src/train_spiking.py --variant v6 --model ratesoft --conds dark \
      --seeds "$s" --steps 1600 --outdir "$OUT" \
      >> "logs/ratesoft_v6_s${s}.log" 2>&1 ) &
done
wait
echo "--- trained ratesoft: $(ls "$OUT"/arm_ratesoft_dark_s*.json 2>/dev/null | wc -l) ---"

echo "=== A1: validity gate ==="
python3 src/gate_blackout.py --indir "$OUT" --variant v6 \
  --pattern "arm_ratesoft_dark_s*.pt" --output runs/gate_ratesoft_v6.json \
  > logs/gate_ratesoft_v6.log 2>&1
echo "ratesoft v6: $(python3 -c "import json;print(json.load(open('runs/gate_ratesoft_v6.json'))['verdict'])")"

echo "=== A1: diagnostic, incl. the validity condition on gradedness ==="
python3 src/surrogate_diag.py --indir "$OUT" --variant v6 \
  --output runs/surrogate_diag_v6.json > logs/surrogate_diag_v6.log 2>&1

python3 src/make_numbers.py --runs runs --output paper/numbers.json \
  --parent-runs ../used_coordinates/runs
echo "CLEAN_TWIN_DONE"
