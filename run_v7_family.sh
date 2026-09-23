#!/bin/bash
# H4 on v7, one process per cell, each under its own gate verdict.
# snn is PASS 8/8; the twin is PARTIAL 7/8 and so runs only under
# --allow-partial, which the output JSON records as a declared deviation.
set -u
cd "$(dirname "$0")"
export PYTHONPATH=src:src/vendor
export OMP_NUM_THREADS=1
for kind in snn rate; do
  G="runs/gate_${kind}_v7.json"
  V=$(python3 -c "import json;print(json.load(open('$G'))['verdict'])")
  EXTRA=""
  [ "$V" = "PARTIAL" ] && EXTRA="--allow-partial"
  if [ "$V" = "FAIL" ]; then echo "REFUSING $kind: $V"; continue; fi
  echo "=== H4 $kind v7 ($V) ==="
  ( python3 src/spike_family.py --indir runs/snn_v7 --gate "$G" $EXTRA \
      --diag runs/surrogate_diag_v7.json --variant v7 --models "$kind" \
      --output "runs/spike_family_v7_${kind}.json" \
      > "logs/family_v7_${kind}.log" 2>&1 ; \
    echo "H4_CELL_DONE $kind" ) &
done
wait
python3 src/make_numbers.py --runs runs --output paper/numbers.json \
  --parent-runs ../used_coordinates/runs
echo "H4_DONE"
