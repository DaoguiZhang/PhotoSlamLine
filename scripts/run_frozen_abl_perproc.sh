#!/usr/bin/env bash
# =============================================================================
# protocol_version: fgm-conv-v2 -- one process per condition.
#
# Each condition runs in its OWN process with its own Gaussian model and its own
# freshly constructed Adam. The two processes share one on-disk initialisation
# checkpoint (PHOTO_SLAM_FROZEN_ABLATION points at the same FROZEN_DIR), so the
# starting state is byte-identical without relying on in-process reuse.
#
# usage: run_frozen_abl_perproc.sh <label> <steps> [scene] [seed] [conds...]
# =============================================================================
set -uo pipefail

LABEL="${1:?usage: run_frozen_abl_perproc.sh <label> <steps> [scene] [seed] [conds...]}"
STEPS="${2:?usage: run_frozen_abl_perproc.sh <label> <steps> [scene] [seed] [conds...]}"
SCENE="${3:-office0}"
SEED="${4:-12345}"
shift 4 2>/dev/null || true
CONDLIST=("$@")
[ ${#CONDLIST[@]} -eq 0 ] && CONDLIST=(B C)

W="${W:-/workspace/code/PSL-fgm-ablation}"
SHARED="${W}/abl_out/${LABEL}_shared/frozen"
mkdir -p "${SHARED}"

echo "protocol_version=fgm-conv-v2  label=${LABEL}  steps=${STEPS}  scene=${SCENE}  seed=${SEED}"
echo "shared frozen dir: ${SHARED}"

for t in "${CONDLIST[@]}"; do
    echo ""
    echo "=========== ${LABEL}: condition ${t} (separate process) ==========="
    CONDS="${t}" \
    FROZEN_DIR="${SHARED}" \
    PROTOCOL_VERSION="fgm-conv-v2" \
    SEQ_SEED="${SEED}" \
    W="${W}" \
    bash "${W}/scripts/run_frozen_abl.sh" "${LABEL}_${t}" "${STEPS}" "${SCENE}" 0
    echo "exit=${?} for condition ${t}"
done

# Publish the shared artifacts under the label so the summariser finds them.
DEST="${W}/abl_out/${LABEL}/frozen"
mkdir -p "${DEST}"
cp -f "${SHARED}"/summary*.csv "${SHARED}"/init_fingerprint.txt \
      "${SHARED}"/*_metrics.csv "${SHARED}"/*_train_views.csv \
      "${SHARED}"/eval_views.txt "${SHARED}"/eval_steps.txt "${DEST}/" 2>/dev/null
echo ""
echo "published to ${DEST}"
echo "summarise with: python3 ${W}/scripts/summarize_frozen_abl.py ${LABEL}"
