#!/usr/bin/env bash
# End-to-end Plucker LBA validation: Mono + RGB-D, office0/office2/room0.
# Each sequence: 1 smoke then 3 runs, LINE_MODE=2 with line-edge counting.
# Proves LocalBundleAdjustmentWithLine_Optimization_Plucker_Reg runs with
# lineVertices>0 (mapLines) and lineEdges>0 (lnEdges) via [LineEdgeDiag].
set -u
cd /workspace/code/Photo-SLAM-L
OUT="${1:-/tmp/plucker_e2e}"
SEQS="office0 office2 room0"
NRUNS="${NRUNS:-3}"
VOCAB=ORB-SLAM3/Vocabulary/ORBvoc.txt
DATAROOT=/workspace/code/SEGS-SLAM/datasets/replica
SUM="$OUT/summary.txt"
mkdir -p "$OUT"
: > "$SUM"

for sensor in mono rgbd; do
  if [ "$sensor" = mono ]; then
    BIN=bin/replica_mono
    CFGDIR=cfg/ORB_SLAM3/Monocular/Replica
    GAUSS=cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml
  else
    BIN=bin/replica_rgbd
    CFGDIR=cfg/ORB_SLAM3/RGB-D/Replica
    GAUSS=cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml
  fi
  for seq in $SEQS; do
    DS="$DATAROOT/$seq"
    ORB_CFG="$CFGDIR/$seq.yaml"
    for i in $(seq 1 $((NRUNS+1))); do
      if [ $i -eq 1 ]; then tag=smoke; else tag=run_$((i-1)); fi
      d="$OUT/${sensor}_${seq}_${tag}"
      mkdir -p "$d"
      env PHOTO_SLAM_LINE_MODE=2 PHOTO_SLAM_LINE_LOOP=0 PHOTO_SLAM_SHADOW_LINE=0 \
        PHOTO_SLAM_SLAM_ONLY=0 PHOTO_SLAM_DEBUG_LINE_EDGES=1 PHOTO_SLAM_DEBUG_MONO_INIT=0 \
        ./$BIN "$VOCAB" "$ORB_CFG" "$GAUSS" "$DS" "$d/" no_viewer > "$d/run.log" 2>&1
      rc=$?
      maps=$(grep -oE 'There are [0-9]+ maps' "$d/run.log" | tail -1)
      lastf=$(grep -oE 'Frame id: [0-9]+' "$d/run.log" | tail -1)
      # Line-edge proof: use max over all LBA calls, plus call count and
      # non-zero-edge call count (tail is unreliable for "did the line LBA run").
      lnEdgeVals=$(grep -oE '\[LineEdgeDiag\] LBA .*lnEdges=[0-9]+' "$d/run.log" | grep -oE 'lnEdges=[0-9]+' | sed 's/lnEdges=//')
      maxLnEdges=0; nLba=0; nLbaNonzero=0
      for v in $lnEdgeVals; do
        nLba=$((nLba+1))
        if [ "$v" -gt "$maxLnEdges" ]; then maxLnEdges=$v; fi
        if [ "$v" -gt 0 ]; then nLbaNonzero=$((nLbaNonzero+1)); fi
      done
      mapLines=$(grep -oE 'mapLines=[0-9]+' "$d/run.log" | grep -oE '[0-9]+' | sort -n | tail -1)
      naninf=$(grep -ciE 'nan|inf' "$d/run.log")
      echo "$sensor $seq $tag EXIT=$rc $lastf $maps maxLnEdges=$maxLnEdges mapLines=${mapLines:-0} lba=$nLba lbaNonzero=$nLbaNonzero naninf=${naninf:-0}" | tee -a "$SUM"
    done
  done
done
echo "=== plucker_e2e done ===" | tee -a "$SUM"
