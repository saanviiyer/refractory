#!/bin/bash
# A1 first (shorter, and answers the architecture-vs-machinery question the v6
# pilot left open), then the v7 environment axis.
set -u
cd "$(dirname "$0")"
./run_clean_twin.sh
./run_pilot.sh v7
echo "ALL_DONE"
