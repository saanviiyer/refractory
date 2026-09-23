#!/bin/bash
# PREREG A5: the untrained control, testing the bounded-code confound.
#
# Random weights, no gradient step, but the same pre-gradient threshold
# calibration the trainer applies -- without it the population never fires and
# the control returns "nothing fits" by construction, which is an artefact
# wearing the reassuring answer's clothes.
#
# The gate is bypassed on the record: an untrained model will not beat frame
# persistence, and refusing to analyse it would defeat the control.
set -u
cd "$(dirname "$0")"
export PYTHONPATH=src:src/vendor
export OMP_NUM_THREADS=1
mkdir -p logs

for kind in snn rate; do
  echo "=== A5 untrained control: $kind, v7, seeds 8-15 ==="
  ( python3 src/spike_family.py --indir runs/conf_v7 --untrained \
      --gate /dev/null --variant v7 --models "$kind" \
      --output "runs/spike_family_untrained_v7_${kind}.json" \
      > "logs/untrained_v7_${kind}.log" 2>&1 ; echo "A5_CELL_DONE $kind" ) &
done
wait

echo "=== A5: score against the trained result ==="
python3 src/confirm_a4.py \
  --snn-family runs/spike_family_untrained_v7_snn.json \
  --rate-family runs/spike_family_untrained_v7_rate.json \
  --snn-gate runs/gate_snn_conf_v7.json \
  --rate-gate runs/gate_rate_conf_v7.json \
  --output runs/a5_untrained_verdict.json
python3 src/make_numbers.py --runs runs --output paper/numbers.json \
  --parent-runs ../used_coordinates/runs
echo "A5_DONE"
