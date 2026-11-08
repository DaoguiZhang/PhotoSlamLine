#!/usr/bin/env bash
# Exit regression for the [EXIT-FIX] work. Three cases, each a separate process:
#
#   A  normal completion      -> exit 0, queue empty, products written, no orphan
#   B  early failure          -> exit 4, no abort/terminate, no orphan
#   C  event audit            -> exit 0 + [EvtAudit] lines for the production
#                                chain, per field, against the closed-form
#                                similarity transform
#
# "No orphan" = after the process is gone, no process with the same binary is
# still alive, and the timeout wrapper did not have to kill anything.
#
# usage: exit_regression.sh [scene-for-C]
set -u
W=/workspace/code/PSL-final
ROOT=/workspace/code/PSL-evtrepl
DATA=$ROOT/export
OUT=$ROOT/exitreg
mkdir -p "$OUT"
SUM=$OUT/summary.csv
: > "$SUM"
echo "case,scene,steps,conds,exit_code,exit_note,orphan,terminate_seen,products_ok,extra" >> "$SUM"

export LD_LIBRARY_PATH=$W/ORB-SLAM3/lib:$W/ORB-SLAM3/Thirdparty/g2o/lib:$W/ORB-SLAM3/Thirdparty/DBoW2/lib:$W/lib
export PHOTO_SLAM_LINE_MODE=2 PHOTO_SLAM_LINE_LOOP=0 PHOTO_SLAM_SHADOW_LINE=0
export PHOTO_SLAM_SLAM_ONLY=0 PHOTO_SLAM_NO_PACING=0 PHOTO_SLAM_SEED=0
export PHOTO_SLAM_DEDUP_VOXEL=1 PHOTO_SLAM_DEDUP_NOTIFY=1

run_case() {   # name scene steps conds replaydir outdir [extra env]
  local name="$1" scene="$2" steps="$3" conds="$4" rdir="$5" odir="$6" extra="${7:-}"
  local BIN ORB GAU SEQ ASSOC
  case "$scene" in
    office0_mono)
      BIN=replica_mono
      ORB=$W/cfg/ORB_SLAM3/Monocular/Replica/office0.yaml
      GAU=$W/cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml
      SEQ=/workspace/code/SEGS-SLAM/datasets/replica/office0; ASSOC="" ;;
    desk3_rgbd)
      BIN=eth3d_rgbd
      ORB=$W/cfg/ORB_SLAM3/RGB-D/ETH3D/eth3d_desk_3.yaml
      GAU=$W/cfg/gaussian_mapper/RGB-D/ETH3D/eth3d_desk_3.yaml
      SEQ=/workspace/code/Photo-SLAM/datasets/eth3d/desk_3
      ASSOC=/workspace/code/Photo-SLAM/datasets/eth3d/desk_3/associated.txt ;;
  esac
  local ARGS=("$W/ORB-SLAM3/Vocabulary/ORBvoc.txt" "$ORB" "$GAU" "$SEQ")
  [ -n "$ASSOC" ] && ARGS+=("$ASSOC")
  ARGS+=("$odir/" "no_viewer")
  local LOG=$OUT/${name}.log
  rm -rf "$odir"; mkdir -p "$odir"
  env PHOTO_SLAM_REPLAY_DIR="$rdir" PHOTO_SLAM_REPLAY_STEPS="$steps" \
      PHOTO_SLAM_REPLAY_CONDS="$conds" $extra \
      timeout -s TERM 3600 "$W/bin/$BIN" "${ARGS[@]}" > "$LOG" 2>&1
  local rc=$?
  local note="normal" orphan=no term=no
  [ "$rc" -eq 124 ] && note="TIMEOUT(124)" && term=yes
  [ "$rc" -eq 134 ] && note="ABORT(134)"
  [ "$rc" -eq 139 ] && note="SIGSEGV(139)"
  pgrep -f "$W/bin/$BIN" > /dev/null 2>&1 && orphan=yes
  grep -q "terminate called" "$LOG" && term=yes
  local prod=no
  local need=$(echo "$conds" | tr ',' '\n' | wc -l)
  local have=0
  for c in $(echo "$conds" | tr ',' ' '); do
    [ -s "$odir/$c/summary.txt" ] && [ -s "$odir/$c/steps.csv" ] && have=$((have+1))
  done
  [ "$have" -eq "$need" ] && prod=yes
  local extra_out
  # NOTE: `grep -c` prints 0 and exits 1 when there is no match; combining it with
  # `|| echo 0` would append a second, bogus line to the CSV. Assign first, then
  # default the (already numeric) value.
  extra_out=$(grep -c "^\[EvtAudit\]" "$LOG" 2>/dev/null)
  extra_out=${extra_out:-0}
  echo "$name,$scene,$steps,\"$conds\",$rc,$note,$orphan,$term,$prod,evtaudit_lines=$extra_out" >> "$SUM"
  echo "[$name] rc=$rc ($note) orphan=$orphan terminate=$term products=$prod audit_lines=$extra_out"
}

# A: normal completion (products + queue + exit 0)
run_case normal_exit office0_mono 150 "A,V,N,VN" "$DATA/office0_mono_s0" "$OUT/out_normal"

# B: early failure (bad replay dir) -> must be a clean non-zero exit, no abort
run_case early_fail office0_mono 150 "A,V,N,VN" "$DATA/DOES_NOT_EXIST" "$OUT/out_fail"

# C: production-chain event audit on a loop (office0, event at step 1691)
run_case audit_loop office0_mono 2000 "A" "$DATA/office0_mono_s0" "$OUT/out_audit_loop" \
         "PHOTO_SLAM_EVT_AUDIT=1 PHOTO_SLAM_EVT_AUDIT_KFS=4 PHOTO_SLAM_EVT_AUDIT_SAMPLES=64"

# C2: production-chain event audit on a map merge (desk3 s1, event at step 2662)
run_case audit_merge desk3_rgbd 2900 "A" "$DATA/desk3_rgbd_s1" "$OUT/out_audit_merge" \
         "PHOTO_SLAM_EVT_AUDIT=1 PHOTO_SLAM_EVT_AUDIT_KFS=4 PHOTO_SLAM_EVT_AUDIT_SAMPLES=64"

echo; cat "$SUM"
