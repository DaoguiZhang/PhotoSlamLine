#!/usr/bin/env bash
# B/P/L/C interleaved batch (3 runs each, single GPU serial).
# Interleaving avoids loading all runs of one mode consecutively.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# (PO LBA) combos: 00=B 10=P 01=L 11=C
for RUN in 1 2 3; do
  for combo in "0 0" "1 0" "0 1" "1 1"; do
    set -- $combo
    echo "############################################################"
    echo "# batch: PO=$1 LBA=$2 RUN=${RUN}"
    echo "############################################################"
    bash "${SCRIPT_DIR}/run_office2_diag2.sh" "$1" "$2" "${RUN}"
  done
done
echo "=== exp2 batch finished ==="
