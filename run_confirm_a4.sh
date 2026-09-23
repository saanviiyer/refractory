#!/bin/bash
# A4: confirmatory test of view-dependence, on fresh seeds 8-15.
#
# The discovery observation came from seeds 0-7 and was reconstructed after the
# fact, because the pre-registered agree statistic references a view that fits
# nothing.  Re-reading those checkpoints would confirm nothing, so this trains
# new ones into a separate directory and leaves the discovery run untouched.
# Everything except the seeds is identical.
set -u
cd "$(dirname "$0")"
export PYTHONPATH=src:src/vendor
export OMP_NUM_THREADS=1
OUT=runs/conf_v7
SEEDS="8 9 10 11 12 13 14 15"
mkdir -p "$OUT" logs

echo "=== A4 S1: fresh seeds, snn and rate, v7, 1600 steps ==="
for kind in snn rate; do
  for s in $SEEDS; do
    ( python3 src/train_spiking.py --variant v7 --model "$kind" --conds dark \
        --seeds "$s" --steps 1600 --outdir "$OUT" \
        >> "logs/conf_${kind}_v7_s${s}.log" 2>&1 ) &
  done
  wait
  echo "--- trained $kind: $(ls "$OUT"/arm_${kind}_dark_s*.json 2>/dev/null | wc -l) ---"
done

echo "=== A4 S2: validity gate, per cell ==="
for kind in snn rate; do
  python3 src/gate_blackout.py --indir "$OUT" --variant v7 \
    --pattern "arm_${kind}_dark_s*.pt" \
    --output "runs/gate_${kind}_conf_v7.json" \
    > "logs/gate_${kind}_conf_v7.log" 2>&1
  echo "$kind conf_v7: $(python3 -c "import json;print(json.load(open('runs/gate_${kind}_conf_v7.json'))['verdict'])")"
done

echo "=== A4 S3: diagnostic ==="
python3 src/surrogate_diag.py --indir "$OUT" --variant v7 \
  --output runs/surrogate_diag_conf_v7.json \
  > logs/surrogate_diag_conf_v7.log 2>&1

echo "=== A4 S4: family detector, each cell under its own gate ==="
for kind in snn rate; do
  G="runs/gate_${kind}_conf_v7.json"
  V=$(python3 -c "import json;print(json.load(open('$G'))['verdict'])")
  if [ "$V" = "FAIL" ]; then
    echo "REFUSING $kind: gate $V. Per PREREG G1 that is this cell's result."
    continue
  fi
  EXTRA=""
  [ "$V" = "PARTIAL" ] && EXTRA="--allow-partial"
  ( python3 src/spike_family.py --indir "$OUT" --gate "$G" $EXTRA \
      --diag runs/surrogate_diag_conf_v7.json --variant v7 --models "$kind" \
      --output "runs/spike_family_conf_v7_${kind}.json" \
      > "logs/family_conf_v7_${kind}.log" 2>&1 ; echo "A4_CELL_DONE $kind" ) &
done
wait

echo "=== A4 S5: score the amendment ==="
python3 src/confirm_a4.py --output runs/a4_verdict.json
python3 src/make_numbers.py --runs runs --output paper/numbers.json \
  --parent-runs ../used_coordinates/runs
echo "A4_DONE"
