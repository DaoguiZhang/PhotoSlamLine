#!/usr/bin/env bash
# Commit the v2b evaluation work on top of 3a99d4b4 and push the eval branch.
set -euo pipefail
cd /workspace/deliv/cabinet-eval
PARENT=3a99d4b466de11a524f48ce8f66498711e04ef44
export GIT_INDEX_FILE=/tmp/ei4
rm -f "$GIT_INDEX_FILE"
git read-tree "$PARENT"
FILES=$(find examples scripts eval cfg -type f \
  \( -name '*.cpp' -o -name '*.py' -o -name '*.sh' -o -name '*.md' \
     -o -name '*.yaml' -o -name 'fr3_cabinet.txt' -o -name 'fr3_large_cabinet.txt' \))
for f in $FILES; do
  b=$(git hash-object -w -- "$f")
  git update-index --add --cacheinfo 100644,"$b","$f"
done
MSG=/tmp/msg_v2b.txt
: > "$MSG"
{
  echo "eval(v2b): explicit arm dispatch, working coverage evaluator, master manifest"
  echo
  echo "Prerequisite fixes for the cross-dataset matrix. No algorithm, threshold,"
  echo "feature count, initialisation or rescue change."
  echo
  echo "* eval_frontend.cpp: EVAL_ARM drives the API (T0 -> TrackRGBD/TrackMonocular,"
  echo "  pure point; P1/P2 -> WithLine). Arm/mode mismatch aborts. [EVAL-DISPATCH]"
  echo "  and [EVAL-TIMING] witnesses per run. replica-mono/rgbd loaders. Rejects a"
  echo "  non-time-ordered association file."
  echo "* gen_assoc.py: sort accepted pairs by RGB timestamp. The greedy pass emitted"
  echo "  them in best-difference order, which made every RGB-D run sleep tens of"
  echo "  seconds per frame and consume frames out of order."
  echo "* matrix.py: MASTER_MANIFEST (492 rows) + resumable serial executor with"
  echo "  per-run directories, real exit codes and build hashes on every row."
  echo "* gen_dataset_configs.py: Replica/ETH3D configs generated from their own"
  echo "  calibration sources, depth factors cross-checked against max(depth)."
  echo "* coverage.py + test_coverage.py: correct-pose coverage and interval logic"
  echo "  reusing eval_tum_se3_v2.py; 9 required cases pass."
} >> "$MSG"
T=$(git write-tree)
C=$(git commit-tree "$T" -p "$PARENT" -F "$MSG")
git update-ref refs/heads/eval/cross-dataset-frontend "$C"
unset GIT_INDEX_FILE
git push -q origin eval/cross-dataset-frontend
git fetch -q origin eval/cross-dataset-frontend
echo "commit=$C"
echo "remote=$(git rev-parse refs/remotes/origin/eval/cross-dataset-frontend)"
echo "fixbranch=$(git rev-parse fix/line-reproj-triangulate-po-reeval)"
git diff --stat "$PARENT" "$C" | tail -2
