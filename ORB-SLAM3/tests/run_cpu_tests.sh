#!/usr/bin/env bash
# Build and run the CPU regression tests for the line/point tracking fixes.
#
# These tests do NOT need a dataset. They exercise:
#   * the line reprojection geometry (Tcw transform, depth/degeneracy guards)
#   * GeometricTools::Triangulate's input convention (plane coords, not a ray)
#   * g2o edge outlier re-evaluation (point side and line side)
#   * the real pose optimisers on a controlled Frame (parity + invariants)
#   * the capture-path gates (missing/unwritable path, no overwrite)
#
# Requirements: OpenCV 4, Eigen3, Sophus (vendored under Thirdparty/), g2o
# (vendored). Adjust OPENCV_DIR if your OpenCV lives elsewhere.
#
#   ./run_cpu_tests.sh [path-to-built-libORB_SLAM3-and-libg2o]
#
# The two optimiser tests link the SHARED library, so build it first:
#   cmake -S . -B build && cmake --build build -j
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"        # .../ORB-SLAM3
LIBDIR="${1:-$ROOT/lib}"
G2ODIR="$ROOT/Thirdparty/g2o/lib"
OPENCV_INC="${OPENCV_INC:-/usr/local/include/opencv4}"
OUT="${TMPDIR:-/tmp}/orb_cpu_tests"
mkdir -p "$OUT"

CXX=${CXX:-g++}
COMMON="-std=c++17 -O2 -w -I $ROOT -I $ROOT/include -I $ROOT/include/CameraModels \
 -I $ROOT/Thirdparty/g2o -I $ROOT/Thirdparty/Sophus -I /usr/include/eigen3 \
 -isystem $OPENCV_INC"
# The library is built with REGISTER_TIMES; the tests must agree or the Frame
# layout differs (ODR mismatch). Keep this in sync with the library build.
ABI="-DCOMPILEDWITHC11 -DREGISTER_TIMES -D_GLIBCXX_USE_CXX11_ABI=1"
LINKLIB="-L $LIBDIR -lORB_SLAM3 -L $ROOT/Thirdparty/DBoW2/lib -lDBoW2 \
 -L $G2ODIR -lg2o -lboost_serialization -lcrypto \
 -L /usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_features2d \
 -lopencv_calib3d -lopencv_flann -Wl,-rpath,$LIBDIR:$G2ODIR"
export LD_LIBRARY_PATH="$LIBDIR:$ROOT/Thirdparty/DBoW2/lib:$G2ODIR:/usr/local/lib"

fail=0
run() { echo "== $1"; shift; if "$@"; then :; else echo "   FAILED"; fail=1; fi; }

$CXX $COMMON "$HERE/test_line_reproj_geometry.cpp" -o "$OUT/t_reproj" || fail=1
run "line reprojection geometry"        "$OUT/t_reproj"

$CXX $COMMON $ABI "$HERE/test_line_triangulate.cpp" -o "$OUT/t_tri" $LINKLIB || fail=1
run "Triangulate input convention"      "$OUT/t_tri"

$CXX $COMMON "$HERE/test_po_capture_paths.cpp" -o "$OUT/t_cap" || fail=1
run "capture path gates"                "$OUT/t_cap"

$CXX $COMMON $ABI "$HERE/test_pose_opt_outlier_reeval.cpp" -o "$OUT/t_reeval" \
     -L "$LIBDIR" -lORB_SLAM3 -L "$G2ODIR" -lg2o -Wl,-rpath,"$LIBDIR:$G2ODIR" || fail=1
run "g2o edge re-evaluation mechanism"  "$OUT/t_reeval"

$CXX $COMMON $ABI "$HERE/test_pose_opt_real.cpp" -o "$OUT/t_real" $LINKLIB || fail=1
run "real pose optimisers (parity)"     "$OUT/t_real" new

echo
if [ "$fail" -eq 0 ]; then echo "ALL CPU TESTS PASSED"; else echo "SOME TESTS FAILED"; fi
exit "$fail"
