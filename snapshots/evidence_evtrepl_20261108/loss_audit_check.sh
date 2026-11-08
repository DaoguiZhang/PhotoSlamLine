#!/usr/bin/env bash
# Loss-audit verification for the "no structural regulariser" final method.
#
# Two runs per scene, same input, same step budget, only the three structural
# weights differ:
#   OFF  cfg_v2/<scene>.nostruct.yaml      (the final L_core configuration)
#   ON   PSL-final/cfg/.../<scene>.yaml    (the historical Full configuration)
#
# The point is the *audit*, not the PSNR: with OFF the structural block must
# never execute (structural_active_steps == 0) and the mapping loss must be
# photometric only; with ON the audit must report the opposite, which proves the
# audit can actually see a participating term.
#
# usage: loss_audit_check.sh
set -u
W=/workspace/code/PSL-final
ROOT=/workspace/code/PSL-evtrepl
DATA=$ROOT/export
OUT=$ROOT/lossaudit
mkdir -p "$OUT"
: > "$OUT/summary.csv"
echo "scene,cfg,steps,exit_code,structural_active,structural_active_steps,coherence_terms,shape_terms,loss_photometric_only" >> "$OUT/summary.csv"

export LD_LIBRARY_PATH=$W/ORB-SLAM3/lib:$W/ORB-SLAM3/Thirdparty/g2o/lib:$W/ORB-SLAM3/Thirdparty/DBoW2/lib:$W/lib
export PHOTO_SLAM_LINE_MODE=2 PHOTO_SLAM_LINE_LOOP=0 PHOTO_SLAM_SHADOW_LINE=0
export PHOTO_SLAM_SLAM_ONLY=0 PHOTO_SLAM_NO_PACING=0 PHOTO_SLAM_SEED=0
export PHOTO_SLAM_DEDUP_VOXEL=0 PHOTO_SLAM_DEDUP_NOTIFY=0
export PHOTO_SLAM_REPLAY_STEPS=200 PHOTO_SLAM_REPLAY_CONDS=A

one() {  # scene bin orb_cfg gauss_cfg seq assoc expdir tag
  local scene=$1 bin=$2 orb=$3 gau=$4 seq=$5 assoc=$6 exp=$7 tag=$8
  local odir=$OUT/out_${scene}_${tag}
  mkdir -p "$odir"
  local args=("$W/bin/$bin" "$W/ORB-SLAM3/Vocabulary/ORBvoc.txt" "$orb" "$gau" "$seq")
  [ -n "$assoc" ] && args+=("$assoc")
  args+=("$odir/" "no_viewer")
  PHOTO_SLAM_REPLAY_DIR="$exp" timeout -s TERM 3600 "${args[@]}" > "$OUT/${scene}_${tag}.log" 2>&1
  local rc=$?
  local sa sas co sh po
  sa=$(sed -n 's/.*\[LossCfg\].*structural_active=\([0-9]*\).*/\1/p' "$OUT/${scene}_${tag}.log" | tail -1)
  sas=$(sed -n 's/.*\[LossAudit\] steps=[0-9]* structural_active_steps=\([0-9]*\).*/\1/p' "$OUT/${scene}_${tag}.log" | tail -1)
  co=$(sed -n 's/.*coherence_terms=\([0-9]*\).*/\1/p' "$OUT/${scene}_${tag}.log" | tail -1)
  sh=$(sed -n 's/.*shape_terms=\([0-9]*\).*/\1/p' "$OUT/${scene}_${tag}.log" | tail -1)
  po=$(sed -n 's/.*loss_is_photometric_only=\([0-9]*\).*/\1/p' "$OUT/${scene}_${tag}.log" | tail -1)
  echo "$scene,$tag,200,$rc,${sa:-NA},${sas:-NA},${co:-NA},${sh:-NA},${po:-NA}" >> "$OUT/summary.csv"
  echo "[$scene $tag] rc=$rc structural_active=$sa active_steps=$sas photometric_only=$po"
}

# office0_mono
one office0_mono replica_mono \
    $W/cfg/ORB_SLAM3/Monocular/Replica/office0.yaml \
    $ROOT/cfg_v2/office0_mono.nostruct.yaml \
    /workspace/code/SEGS-SLAM/datasets/replica/office0 "" \
    "$DATA/office0_mono_s0" OFF
one office0_mono replica_mono \
    $W/cfg/ORB_SLAM3/Monocular/Replica/office0.yaml \
    $W/cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml \
    /workspace/code/SEGS-SLAM/datasets/replica/office0 "" \
    "$DATA/office0_mono_s0" ON

# desk3_rgbd
one desk3_rgbd eth3d_rgbd \
    $W/cfg/ORB_SLAM3/RGB-D/ETH3D/eth3d_desk_3.yaml \
    $ROOT/cfg_v2/desk3_rgbd.nostruct.yaml \
    /workspace/code/Photo-SLAM/datasets/eth3d/desk_3 \
    /workspace/code/Photo-SLAM/datasets/eth3d/desk_3/associated.txt \
    "$DATA/desk3_rgbd_s1" OFF
one desk3_rgbd eth3d_rgbd \
    $W/cfg/ORB_SLAM3/RGB-D/ETH3D/eth3d_desk_3.yaml \
    $W/cfg/gaussian_mapper/RGB-D/ETH3D/eth3d_desk_3.yaml \
    /workspace/code/Photo-SLAM/datasets/eth3d/desk_3 \
    /workspace/code/Photo-SLAM/datasets/eth3d/desk_3/associated.txt \
    "$DATA/desk3_rgbd_s1" ON

echo; cat "$OUT/summary.csv"
