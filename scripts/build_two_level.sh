#!/usr/bin/env bash
# Unified build for Photo-SLAM-L.
#
# ORB-SLAM3 is a SEPARATE CMake project and the top level only links its
# prebuilt .so. Building only the top level after a shared-header change leaves
# libORB_SLAM3.so compiled against a different type layout than
# libgaussian_mapper.so (this produced the TUM KF2 free(): invalid pointer).
#
# Always use this script (or the exact same two commands) after touching:
#   ORB-SLAM3/include/*.h, ORB-SLAM3/src/*.cc, or shared MappingOperation code.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[build] 1/2 ORB-SLAM3"
cmake --build ORB-SLAM3/build -j8

echo "[build] 2/2 top level"
cmake --build build -j8

echo "[verify] runtime library resolution for bin/tum_rgbd"
ldd bin/tum_rgbd | grep -E 'libORB_SLAM3|libgaussian_mapper|libgaussian_viewer' || true

echo "[verify] MappingOperation destructor layout across shared libraries"
sizes="$(
  for so in ORB-SLAM3/lib/libORB_SLAM3.so lib/libgaussian_mapper.so; do
    size=$(nm -C -S --defined-only "$so" 2>/dev/null \
      | grep -m1 'MappingOperation::~MappingOperation' | awk '{print $2}')
    echo "${size:-missing} $so"
  done
)"
echo "$sizes"
unique=$(echo "$sizes" | awk '{print $1}' | sort -u | wc -l)
if [ "$unique" -ne 1 ]; then
  echo "[verify] ERROR: MappingOperation layout differs between shared libraries" >&2
  exit 1
fi
echo "[verify] OK: both shared libraries use the same MappingOperation layout"
