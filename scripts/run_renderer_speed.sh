(
set -euo pipefail
cd /workspace/code/PSL-perf

dataset=/workspace/code/SEGS-SLAM/datasets/replica/office0
orb_cfg=cfg/ORB_SLAM3/RGB-D/Replica/office0.yaml
gauss_cfg=results/quad_office0/config/gaussian_quad.yaml

test -x bin/replica_rgbd
test -x /usr/bin/time
test -d "$dataset"
test -f "$orb_cfg"
test -f "$gauss_cfg"

mkdir -p results
speed_out=$(mktemp -d "$PWD/results/renderer_speed_XXXXXXXX")
mkdir -p "$speed_out/logs"

git rev-parse HEAD > "$speed_out/commit.txt"
git status --short > "$speed_out/source_status.txt"
cp "$gauss_cfg" "$speed_out/gaussian_config.yaml"

sha256sum \
  bin/replica_rgbd \
  lib/libgaussian_mapper.so \
  ORB-SLAM3/lib/libORB_SLAM3.so \
  "$gauss_cfg" > "$speed_out/checksums.txt"

for round in 0 1 2; do
  order=(A C)
  if (( round == 1 )); then
    order=(C A)
  fi

  for backend in "${order[@]}"; do
    run_out="$speed_out/$backend/replica_rgbd_$round/office0"
    run_log="$speed_out/logs/${backend}_r${round}.log"
    time_log="$speed_out/logs/${backend}_r${round}.time"
    mkdir -p "$run_out"

    printf 'Running backend=%s round=%s\n' "$backend" "$round"

    if /usr/bin/time \
      -f 'wall_seconds=%e\nexit_code=%x' \
      -o "$time_log" \
      env \
        -u CUDA_LAUNCH_BLOCKING \
        -u PHOTO_SLAM_RASTER_AGG_SELFCHECK \
        -u PHOTO_SLAM_RASTER_PROF \
        -u PHOTO_SLAM_GPU_PROF \
        PHOTO_SLAM_RASTER_BACKEND="$backend" \
        PHOTO_SLAM_LINE_MODE=2 \
        PHOTO_SLAM_GBA_APPLY_CORRECTION=0 \
        PHOTO_SLAM_APPEND_STEPS=0 \
      bin/replica_rgbd \
        ORB-SLAM3/Vocabulary/ORBvoc.txt \
        "$orb_cfg" \
        "$gauss_cfg" \
        "$dataset" \
        "$run_out" \
        no_viewer > "$run_log" 2>&1
    then
      printf 'Completed: backend=%s round=%s\n' "$backend" "$round"
    else
      run_code=$?
      printf 'FAILED: exit=%s log=%s\n' "$run_code" "$run_log"
      exit "$run_code"
    fi
  done
done

printf '\nResults directory: %s\n' "$speed_out"
rg -n 'wall_seconds|exit_code' "$speed_out/logs"/*.time
)
