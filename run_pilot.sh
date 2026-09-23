#!/bin/bash
# Train, gate, refuse on failure, and only then analyse.
#
# The gate sits between training and analysis and is not optional.  The parent
# lost a 96-model sweep and a day of belief to a term that contributed no
# gradient while the loss curve looked healthy; a spiking model has the same
# failure available to it through a surrogate that never reaches the operating
# margin.  No statistic of the term catches either.  Prior-path prediction
# against frame persistence does.
#
# Usage: ./run_pilot.sh [v6|v7] [snn|snn_wide] [seeds]
set -u
cd "$(dirname "$0")"
export PYTHONPATH=src:src/vendor
export OMP_NUM_THREADS=1

VARIANT="${1:-v6}"
MODEL="${2:-snn}"
TWIN=$([ "$MODEL" = "snn_wide" ] && echo rate_wide || echo rate)
SEEDS="${3:-0 1 2 3 4 5 6 7}"
STEPS=1600
OUT="runs/${MODEL}_${VARIANT}"
mkdir -p "$OUT" logs

echo "=== S0: the gate must be calibrated before it is trusted ==="
if [ ! -f runs/gate_calibration.json ]; then
  python3 src/gate_blackout.py --calibrate --output runs/gate_calibration.json \
    > logs/gate_calibration.log 2>&1
fi
CAL=$(python3 -c "import json;print(json.load(open('runs/gate_calibration.json'))['calibrated'])")
echo "calibrated=$CAL"
if [ "$CAL" != "True" ]; then
  echo "REFUSING: the gate does not reproduce the parent's known verdicts."
  echo "See runs/gate_calibration.json. Its verdict on a new architecture"
  echo "would be worthless in both directions."
  exit 1
fi

echo "=== S1: train $MODEL and its twin $TWIN on $VARIANT, $STEPS steps ==="
for kind in "$MODEL" "$TWIN"; do
  for s in $SEEDS; do
    ( python3 src/train_spiking.py --variant "$VARIANT" --model "$kind" \
        --conds dark --seeds "$s" --steps "$STEPS" --outdir "$OUT" \
        >> "logs/${kind}_${VARIANT}_s${s}.log" 2>&1 ) &
  done
  wait
  echo "--- trained $kind: $(ls "$OUT"/arm_${kind}_dark_s*.json 2>/dev/null | wc -l) ---"
done

echo "=== S2: validity gate ==="
for kind in "$MODEL" "$TWIN"; do
  python3 src/gate_blackout.py --indir "$OUT" --variant "$VARIANT" \
    --pattern "arm_${kind}_dark_s*.pt" \
    --output "runs/gate_${kind}_${VARIANT}.json" \
    > "logs/gate_${kind}_${VARIANT}.log" 2>&1
  V=$(python3 -c "import json;print(json.load(open('runs/gate_${kind}_${VARIANT}.json'))['verdict'])")
  echo "$kind $VARIANT: $V"
done

echo "=== S3: surrogate diagnostic (not a gate) ==="
python3 src/surrogate_diag.py --indir "$OUT" --variant "$VARIANT" \
  --output "runs/surrogate_diag_${VARIANT}.json" \
  > "logs/surrogate_diag_${VARIANT}.log" 2>&1

echo "=== S4: the criterion, across state views ==="
# PREREG G1 gates per cell -- per (model, variant) -- so each model is analysed
# under *its own* verdict.  An earlier version passed one --gate while naming
# both models to --models, which would have analysed the twin under the primary
# model's verdict; on v7 that mattered, because snn passed 8/8 while the twin
# came in at 7/8.  A PARTIAL cell is analysed only under --allow-partial, which
# is recorded in the output JSON as the declared deviation it is.
for kind in "$MODEL" "$TWIN"; do
  G="runs/gate_${kind}_${VARIANT}.json"
  V=$(python3 -c "import json;print(json.load(open('$G'))['verdict'])")
  EXTRA=""
  if [ "$V" = "FAIL" ]; then
    echo "REFUSING to analyse $kind on $VARIANT: no seed beats frame"
    echo "persistence during a blackout. See $G."
    echo "Per PREREG.md G1 this failure IS the result for this cell."
    continue
  elif [ "$V" = "PARTIAL" ]; then
    echo "$kind $VARIANT is PARTIAL -- analysing under --allow-partial,"
    echo "recorded in the output JSON as a declared deviation."
    EXTRA="--allow-partial"
  fi
  if [ "$VARIANT" = "v7" ]; then
    python3 src/spike_family.py --indir "$OUT" --gate "$G" $EXTRA \
      --diag "runs/surrogate_diag_${VARIANT}.json" \
      --variant "$VARIANT" --models "$kind" \
      --output "runs/spike_family_${VARIANT}_${kind}.json" \
      > "logs/family_${VARIANT}_${kind}.log" 2>&1
  else
    python3 src/spike_torus.py --indir "$OUT" --gate "$G" $EXTRA \
      --diag "runs/surrogate_diag_${VARIANT}.json" \
      --variant "$VARIANT" --mode search --conds dark --models "$kind" \
      --output "runs/spike_torus_${VARIANT}_${kind}.json" \
      > "logs/torus_${VARIANT}_${kind}.log" 2>&1
  fi
  echo "--- analysed $kind ($V) ---"
done

echo "=== S8: macros ==="
python3 src/make_numbers.py --runs runs --output paper/numbers.json
echo "PILOT_DONE $MODEL $VARIANT"
