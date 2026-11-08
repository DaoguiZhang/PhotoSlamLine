#!/usr/bin/env bash
# Small-sample acceptance for the locked full-GPU-mapping version.
# Every case uses the SAME locked binary; only the Gaussian YAML differs
# (Full = official weights, OFF = the same YAML with the three effective
# structural-loss weights zeroed).
set -uo pipefail

EXP=/workspace/code/PSL-fullmap-exp-v1
SRC=/workspace/code/PSL-fullmap-lock-v1
BIN="$SRC/bin"
VOC="$SRC/ORB-SLAM3/Vocabulary/ORBvoc.txt"
OUT="$EXP/small"
export LD_LIBRARY_PATH="$SRC/ORB-SLAM3/lib:$SRC/ORB-SLAM3/Thirdparty/g2o/lib:$SRC/ORB-SLAM3/Thirdparty/DBoW2/lib:$SRC/lib"

export PHOTO_SLAM_LINE_MODE=2
export PHOTO_SLAM_LINE_LOOP=0
export PHOTO_SLAM_SHADOW_LINE=0
export PHOTO_SLAM_SLAM_ONLY=0
export PHOTO_SLAM_NO_PACING=0

record_meta() {
    local d="$1" name="$2" bin="$3" orb="$4" gau="$5" seq="$6" assoc="$7" cond="$8"
    {
        echo "case=$name"
        echo "condition=$cond"
        echo "binary=$bin"
        echo "binary_sha256=$(sha256sum "$bin" | cut -d' ' -f1)"
        echo "libORB_SLAM3=$SRC/ORB-SLAM3/lib/libORB_SLAM3.so"
        echo "libORB_SLAM3_sha256=$(sha256sum "$SRC/ORB-SLAM3/lib/libORB_SLAM3.so" | cut -d' ' -f1)"
        echo "libgaussian_mapper=$SRC/lib/libgaussian_mapper.so"
        echo "libgaussian_mapper_sha256=$(sha256sum "$SRC/lib/libgaussian_mapper.so" | cut -d' ' -f1)"
        echo "orb_config=$orb"
        echo "orb_config_sha256=$(sha256sum "$orb" | cut -d' ' -f1)"
        echo "gaussian_config=$gau"
        echo "gaussian_config_sha256=$(sha256sum "$gau" | cut -d' ' -f1)"
        echo "sequence=$seq"
        echo "association=${assoc:-none}"
        echo "line_mode=$PHOTO_SLAM_LINE_MODE line_loop=$PHOTO_SLAM_LINE_LOOP shadow_line=$PHOTO_SLAM_SHADOW_LINE slam_only=$PHOTO_SLAM_SLAM_ONLY"
        echo "seed=${PHOTO_SLAM_SEED:-unset}"
        echo "debug_line_loss=${PHOTO_SLAM_DEBUG_LINE_LOSS:-unset}"
        echo "ld_library_path=$LD_LIBRARY_PATH"
    } > "$d/run.meta"
}

run_case() {
    local name="$1" bin="$2" orb="$3" gau="$4" seq="$5" assoc="$6" cond="$7" seed="$8" dbg="$9"
    local d="$OUT/${name}_${cond}"
    rm -rf "$d"; mkdir -p "$d"
    export PHOTO_SLAM_SEED="$seed"
    if [ "$dbg" = "1" ]; then export PHOTO_SLAM_DEBUG_LINE_LOSS=1; else unset PHOTO_SLAM_DEBUG_LINE_LOSS; fi
    record_meta "$d" "$name" "$bin" "$orb" "$gau" "$seq" "$assoc" "$cond"
    cp -- "$orb" "$d/orb_settings.used.yaml"
    cp -- "$gau" "$d/gaussian_settings.used.yaml"

    local -a argv
    if [ -n "$assoc" ]; then
        argv=("$bin" "$VOC" "$orb" "$gau" "$seq" "$assoc" "$d/" no_viewer)
    else
        argv=("$bin" "$VOC" "$orb" "$gau" "$seq" "$d/" no_viewer)
    fi
    local t0=$SECONDS
    timeout --signal=TERM --kill-after=60 2400 "${argv[@]}" > "$d/stdout.log" 2>&1
    local rc=$?
    echo "exit=$rc wall_s=$((SECONDS-t0))" >> "$d/run.meta"
    echo "[$name/$cond] exit=$rc wall_s=$((SECONDS-t0))"
}

# 1) Paired Full/OFF on Replica Mono office0 (200 frames), structural-loss
#    diagnostics ON so the effect of the weights is directly observable.
RO="$SRC/cfg/ORB_SLAM3/Monocular/Replica/office0.yaml"
RG="$SRC/cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml"
RGO="$EXP/configs/off/replica_mono.off.yaml"
run_case replica_office0_mono_200 "$BIN/replica_mono" "$RO" "$RG"  "$EXP/inputs/replica_office0_mono_200" "" full 0 1
run_case replica_office0_mono_200 "$BIN/replica_mono" "$RO" "$RGO" "$EXP/inputs/replica_office0_mono_200" "" off  0 1

# 2) TUM RGB-D fr3 long office household (200 frames), paired.
TO="$SRC/cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_long_office_household.yaml"
TG="$SRC/cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml"
TGO="$EXP/configs/off/tum_rgbd.off.yaml"
TSEQ=/workspace/code/SEGS-SLAM/datasets/tum/rgbd_dataset_freiburg3_long_office_household
TASSO="$EXP/inputs/tum_fr3_rgbd_assoc_200.txt"
run_case tum_fr3_rgbd_200 "$BIN/tum_rgbd" "$TO" "$TG"  "$TSEQ" "$TASSO" full 0 0
run_case tum_fr3_rgbd_200 "$BIN/tum_rgbd" "$TO" "$TGO" "$TSEQ" "$TASSO" off  0 0

# 3) ETH3D RGB-D desk_3 (200 frames), Full.
EO="$SRC/cfg/ORB_SLAM3/RGB-D/ETH3D/eth3d_desk_3.yaml"
EG="$SRC/cfg/gaussian_mapper/RGB-D/ETH3D/eth3d_desk_3.yaml"
ESEQ=/workspace/code/Photo-SLAM/datasets/eth3d/desk_3
EASSO="$EXP/inputs/eth3d_desk3_rgbd_assoc_200.txt"
run_case eth3d_desk3_rgbd_200 "$BIN/eth3d_rgbd" "$EO" "$EG" "$ESEQ" "$EASSO" full 0 0

# 4) ETH3D Mono table_7 (600 frames), Full. The ETH3D mono Gaussian config
#    requires Mapper.min_num_initial_map_kfs=28; a 200-frame view reaches only
#    ~21 KFs, so the mapper correctly reports an empty model and skips export.
#    600 frames reach the threshold and produce a non-empty PLY.
EMO="$SRC/cfg/ORB_SLAM3/Monocular/ETH3D/eth3d_table_7.yaml"
EMG="$SRC/cfg/gaussian_mapper/Monocular/ETH3D/eth3d_table_7.yaml"
run_case eth3d_table7_mono_600 "$BIN/eth3d_mono" "$EMO" "$EMG" "$EXP/inputs/eth3d_table7_mono_600" "" full 0 0
