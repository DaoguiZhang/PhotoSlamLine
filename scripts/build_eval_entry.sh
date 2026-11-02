#!/usr/bin/env bash
# Build the unified SLAM-only evaluation entry (examples/eval_frontend.cpp).
# Deliberately links ONLY libORB_SLAM3 + OpenCV: the Gaussian mapper, viewer,
# simple-knn, CUDA rasterizer, imgui and torch are filtered out of the recorded
# link line so the front-end evaluation cannot accidentally depend on them.
set -euo pipefail

TREE=$(cd "$(dirname "$0")/.." && pwd)
OBR=$TREE/ORB-SLAM3
REF=/workspace/code/Photo-SLAM-L
BUILD=/tmp/build_eval_entry
LIB=${LIBDIR:-$OBR/lib}
mkdir -p "$BUILD" "$TREE/bin"

flags="$REF/build/CMakeFiles/tum_mono.dir/flags.make"
link="$REF/build/CMakeFiles/tum_mono.dir/link.txt"

inc=$(sed -n 's/^CXX_INCLUDES = //p' "$flags" | tr -d '\\\n' | sed "s#$REF#${TREE}#g")
cxxflags=$(sed -n 's/^CXX_FLAGS = //p' "$flags" | tr -d '\\\n')
defs=$(sed -n 's/^CXX_DEFINES = //p' "$flags" | tr -d '\\\n')

echo "=== compile eval_frontend"
# shellcheck disable=SC2086
/usr/bin/g++ $defs $inc $cxxflags -o "$BUILD/eval_frontend.o" -c "$TREE/examples/eval_frontend.cpp"

echo "=== link eval_frontend"
tr ' ' '\n' < "$link" \
  | grep -v -E 'libgaussian_mapper|libgaussian_viewer|libsimple_knn|libcuda_rasterizer|libimgui|torch|cudart|nvrtc|libc10|glfw|libGLX|libOpenGL|tum_mono\.cpp\.o' \
  | grep -v -E '^$|^-o$|^\.\./bin/|^/usr/bin/g\+\+$' \
  | sed -e "s#^\.\./ORB-SLAM3/lib/libORB_SLAM3\.so#$LIB/libORB_SLAM3.so#" \
        -e "s#$REF/ORB-SLAM3/lib/libORB_SLAM3.so#$LIB/libORB_SLAM3.so#g" \
  > "$BUILD/link_args.txt"

{
  echo -n "/usr/bin/g++ $BUILD/eval_frontend.o -o $TREE/bin/eval_frontend -Wl,-rpath,$LIB "
  tr '\n' ' ' < "$BUILD/link_args.txt"
  echo
} > "$BUILD/link.txt"

( cd "$TREE" && bash "$BUILD/link.txt" )

echo "=== done"
ls -l "$TREE/bin/eval_frontend"
sha256sum "$TREE/bin/eval_frontend"
echo "--- NEEDED"
objdump -p "$TREE/bin/eval_frontend" | grep NEEDED
