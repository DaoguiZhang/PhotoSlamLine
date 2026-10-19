#!/usr/bin/env bash
# Run the full A/B/C ablation batch (3 runs each) sequentially.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for MODE in 0 1 2; do
  for RUN in 1 2 3; do
    echo "############################################################"
    echo "# batch: MODE=${MODE} RUN=${RUN}"
    echo "############################################################"
    bash "${SCRIPT_DIR}/run_office2_diagnosis.sh" "${MODE}" "${RUN}"
  done
done

echo "=== batch finished ==="
python3 "${SCRIPT_DIR}/summarize_office2_diagnosis.py"
