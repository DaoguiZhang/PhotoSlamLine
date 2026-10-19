#!/usr/bin/env bash
# Build an ASan+UBSan instrumented libORB_SLAM3.so (and instrumented g2o) in a
# separate build dir and stage them under /tmp. The good libraries under
# ORB-SLAM3/lib and ORB-SLAM3/Thirdparty/g2o/lib are backed up and restored so
# the working tree stays pristine.
#
# Output:
#   /tmp/libORB_SLAM3_asan.so
#   /tmp/libg2o_asan.so
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OS3="${PROJECT_ROOT}/ORB-SLAM3"
CMAKE="${OS3}/CMakeLists.txt"
BUILD_DIR="${OS3}/build_asan"
G2O_LIB="${OS3}/Thirdparty/g2o/lib/libg2o.so"

# 1. Back up pristine artifacts.
cp -v "${CMAKE}" /tmp/ORB_SLAM3_CMakeLists.txt.bak
G2O_BAK=""
if [ -f "${G2O_LIB}" ]; then
    G2O_BAK="/tmp/libg2o_good_build_$(md5sum "${G2O_LIB}" | awk '{print $1}').so"
    cp -v "${G2O_LIB}" "${G2O_BAK}"
fi

# 2. Patch CMakeLists: sanitizer flags + separate SLAM output dir. Keep
#    add_subdirectory(Thirdparty/g2o) so g2o is instrumented too.
sed -i \
  -e 's#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}   -O3")#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer")#' \
  -e 's#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -O3")#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer")#' \
  -e 's#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib_asan)#' \
  "${CMAKE}"

# 3. Configure + build.
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
cmake .. \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_FLAGS="-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=address,undefined" \
  > cmake_configure.log 2>&1

make -j8 > make.log 2>&1 || { tail -40 make.log; exit 1; }

# 4. Stage instrumented libraries.
cp -v "${OS3}/lib_asan/libORB_SLAM3.so" /tmp/libORB_SLAM3_asan.so
cp -v "${OS3}/Thirdparty/g2o/lib/libg2o.so" /tmp/libg2o_asan.so

# 5. Restore pristine CMakeLists + good g2o.
cp -v /tmp/ORB_SLAM3_CMakeLists.txt.bak "${CMAKE}"
if [ -n "${G2O_BAK}" ] && [ -f "${G2O_BAK}" ]; then
    cp -v "${G2O_BAK}" "${G2O_LIB}"
fi

echo "ASan/UBSan libraries staged:"
md5sum /tmp/libORB_SLAM3_asan.so /tmp/libg2o_asan.so
