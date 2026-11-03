#!/usr/bin/env bash
# Isolated, flag-consistent builds of g2o + ORB-SLAM3 + entry.
# Two independent trees: Release and ASan. Nothing is reused from older build
# directories. Purpose: decide whether the "free on non-malloc-ed address" seen
# earlier is a property of a MIXED build (ASan ORB-SLAM3 + non-ASan g2o) or a
# real defect.
set -euo pipefail

TREE=/workspace/deliv/cabinet-eval
OBR=$TREE/ORB-SLAM3
ROOT=/workspace/eval_builds/v1
COMMON_DEFS="-DCOMPILEDWITHC11 -DORB_SLAM3_EXPORTS -DREGISTER_TIMES -D_GLIBCXX_USE_CXX11_ABI=1"
BASE="-O3 -DNDEBUG -fPIC -w -std=c++17 -pthread -Wno-deprecated-declarations"
ASAN="-O1 -DNDEBUG -fPIC -w -std=c++17 -pthread -fsanitize=address -fno-omit-frame-pointer -Wno-deprecated-declarations"

mkdir -p "$ROOT"
report() { echo "### $*"; }

echo "=== toolchain / eigen identity"
/usr/bin/g++ --version | head -1
echo "eigen_include=$(ls -d /usr/include/eigen3)"
grep -h "define EIGEN_WORLD_VERSION\|define EIGEN_MAJOR_VERSION\|define EIGEN_MINOR_VERSION" \
  /usr/include/eigen3/Eigen/src/Core/util/Macros.h

# ---------------------------------------------------------------- g2o ------
build_g2o() {  # $1 = tag, $2 = cxx flags
  local tag=$1 flags=$2 src=$ROOT/g2o_$1 src_src=$OBR/Thirdparty/g2o
  rm -rf "$src"; mkdir -p "$src"
  cp -a "$src_src"/. "$src"/
  rm -rf "$src/build"
  ( cd "$src" && mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="$flags" >/dev/null && \
    make -j12 >/dev/null 2>&1 )
  report "g2o($tag) flags: $flags"
  sha256sum "$src/lib/libg2o.so" | sed 's/^/  /'
}

# ------------------------------------------------------------ ORB-SLAM3 ----
build_orb() {  # $1 = tag, $2 = flags, $3 = g2o lib dir, $4 = dbow lib dir
  local tag=$1 flags=$2 g2o=$3 dbow=$4
  local B=$ROOT/obj_$tag OUT=$ROOT/lib_$tag
  rm -rf "$B" "$OUT"; mkdir -p "$B" "$OUT"
  local INC="-I$OBR -I$OBR/include -I$OBR/include/CameraModels -I$OBR/Thirdparty/Sophus -I/usr/include/eigen3 -isystem /usr/local/include/opencv4"
  ls "$OBR"/src/*.cc "$OBR"/src/*.cpp "$OBR"/src/CameraModels/*.cpp | \
    xargs -P 12 -I{} sh -c "/usr/bin/g++ $COMMON_DEFS $INC $flags -c '{}' -o $B/\$(basename '{}' | sed 's/\.[^.]*\$//').o"
  local CV; CV=$(ls /usr/local/lib/libopencv_*.so.4.10.0 | tr '\n' ' ')
  local SAN=""; case "$flags" in *sanitize*) SAN="-fsanitize=address";; esac
  /usr/bin/g++ -shared $SAN -Wl,-soname,libORB_SLAM3.so -o "$OUT/libORB_SLAM3.so" \
    "$B"/*.o "$dbow/lib/libDBoW2.so" "$g2o/lib/libg2o.so" \
    -lboost_serialization -lcrypto \
    -Wl,-rpath,"$dbow/lib":"$g2o/lib" $CV -pthread
  report "ORB-SLAM3($tag) flags: $flags"
  sha256sum "$OUT/libORB_SLAM3.so" | sed 's/^/  /'
  echo "  g2o_loaded_by_needed=$(objdump -p "$OUT/libORB_SLAM3.so" | awk '/NEEDED/{print $2}' | tr '\n' ' ')"
}

build_entry() {  # $1 = tag, $2 = flags, $3 = lib dir
  local tag=$1 flags=$2 lib=$3
  local B=$ROOT/entry_$tag; rm -rf "$B"; mkdir -p "$B"
  local INC="-I$TREE -I$OBR -I$OBR/include -I$OBR/include/CameraModels -I$OBR/Thirdparty/Sophus -I/usr/include/eigen3 -isystem /usr/local/include/opencv4"
  local SAN=""; case "$flags" in *sanitize*) SAN="-fsanitize=address";; esac
  /usr/bin/g++ $COMMON_DEFS $INC $flags -c "$TREE/examples/eval_frontend.cpp" -o "$B/eval_frontend.o"
  local CV; CV=$(ls /usr/local/lib/libopencv_*.so.4.10.0 | tr '\n' ' ')
  /usr/bin/g++ $SAN "$B/eval_frontend.o" -o "$B/eval_frontend" -Wl,-rpath,"$lib" "$lib/libORB_SLAM3.so" $CV -pthread
  report "entry($tag)"
  sha256sum "$B/eval_frontend" | sed 's/^/  /'
  echo "  runtime libs:"; ldd "$B/eval_frontend" | grep -E "ORB_SLAM3|g2o" | sed 's/^/    /'
}

echo; echo "=== DBoW2 (shared, unchanged, no Eigen in its ABI path)"
DB=$OBR/Thirdparty/DBoW2
ls "$DB/lib/libDBoW2.so" >/dev/null && sha256sum "$DB/lib/libDBoW2.so" | sed 's/^/  /'

echo; echo "=== Release tree"
build_g2o rel "$BASE"
build_orb rel "$BASE" "$ROOT/g2o_rel" "$DB"
build_entry rel "$BASE" "$ROOT/lib_rel"

echo; echo "=== ASan tree (g2o, ORB-SLAM3 and entry ALL instrumented)"
build_g2o asan "$ASAN"
build_orb asan "$ASAN" "$ROOT/g2o_asan" "$DB"
build_entry asan "$ASAN" "$ROOT/lib_asan"

echo; echo "=== DONE"
