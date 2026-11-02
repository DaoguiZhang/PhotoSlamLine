#!/usr/bin/env bash
# Independent build of ORB-SLAM3 + the two SLAM-only examples from the DELIVERY
# tree (commit d3214194 or later). No Gaussian/CUDA target is built and no
# mapper library is linked, so the examples cannot start the backend.
#
# Variant selection mirrors the fix branch:
#   VARIANT=new (default) : point+line outlier re-evaluation compiled IN
#   VARIANT=old           : -DPO_LINE_NO_OUTLIER_REEVAL -DPO_LINE_NO_LINE_OUTLIER_REEVAL
set -euo pipefail

TREE=$(cd "$(dirname "$0")/.." && pwd)      # .../cabinet-eval
OBR=$TREE/ORB-SLAM3
VARIANT="${VARIANT:-new}"
if [ "$VARIANT" = "old" ]; then
  EXTRA="-DPO_LINE_NO_OUTLIER_REEVAL -DPO_LINE_NO_LINE_OUTLIER_REEVAL"
  OUTDIR=$OBR/lib_old
  BUILD=/tmp/build_eval_old
else
  EXTRA=""
  OUTDIR=$OBR/lib
  BUILD=/tmp/build_eval_new
fi
mkdir -p "$BUILD/obj" "$OUTDIR" "$TREE/bin"

DEFS="-DCOMPILEDWITHC11 -DORB_SLAM3_EXPORTS -DREGISTER_TIMES -D_GLIBCXX_USE_CXX11_ABI=1 $EXTRA"
INC="-I$OBR -I$OBR/include -I$OBR/include/CameraModels -I$OBR/Thirdparty/Sophus \
-I$OBR/Thirdparty/g2o -I/usr/include/eigen3 -isystem /usr/local/include/opencv4"
FLAGS="-Wno-deprecated-declarations -O3 -O2 -g -DNDEBUG -fPIC -w -std=c++17 -pthread"

echo "=== compiling $(ls $OBR/src/*.cc $OBR/src/*.cpp $OBR/src/CameraModels/*.cpp | wc -l) translation units"
ls "$OBR"/src/*.cc "$OBR"/src/*.cpp "$OBR"/src/CameraModels/*.cpp | xargs -P 12 -I{} sh -c '
  src="{}"; obj="'"$BUILD"'/obj/$(basename "${src%.*}").o"
  /usr/bin/g++ '"$DEFS"' '"$INC"' '"$FLAGS"' -c "$src" -o "$obj"
'

CVLIBS=$(ls /usr/local/lib/libopencv_*.so.4.10.0 | tr '\n' ' ')
echo "=== linking libORB_SLAM3.so"
/usr/bin/g++ -shared -Wl,-soname,libORB_SLAM3.so -o "$OUTDIR/libORB_SLAM3.so" \
  "$BUILD"/obj/*.o "$OBR"/Thirdparty/DBoW2/lib/libDBoW2.so \
  "$OBR"/Thirdparty/g2o/lib/libg2o.so -lboost_serialization -lcrypto \
  -Wl,-rpath,"$OBR/Thirdparty/DBoW2/lib":"$OBR/Thirdparty/g2o/lib" \
  $CVLIBS -pthread
echo "variant=$VARIANT defs=$EXTRA"
sha256sum "$OUTDIR/libORB_SLAM3.so"
