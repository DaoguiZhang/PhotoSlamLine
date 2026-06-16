cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(photo-slam LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
    # set(CMAKE_BUILD_TYPE Debug)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

# =========================================================
# 1. 容器环境防坑配置 (从 SEGS-SLAM 继承)
# =========================================================
# 强制开启 C++11 ABI 以兼容容器内的 PyTorch
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

# 屏蔽第三方库(Eigen/g2o)在C++17下的废弃警告，保持终端干净
add_compile_options(-Wno-deprecated-declarations)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

# 全局设置 RTX 5090 的 CUDA 架构
set(CMAKE_CUDA_ARCHITECTURES 120)

set(CAFFE2_USE_CUDNN on)

# =========================================================
# 2. 依赖查找
# =========================================================
# 移除原版写死的宿主机路径，让 CMake 在容器内自动寻找 Torch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -pthread")

find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV 4 REQUIRED)

# JsonCpp 修复：使用 pkg-config (与 SEGS-SLAM 同样的处理)
find_package(PkgConfig REQUIRED)
pkg_check_modules(jsoncpp REQUIRED jsoncpp)

find_package(OpenGL REQUIRED)
find_package(glm REQUIRED)
find_package(glfw3 REQUIRED)
find_package(realsense2 QUIET)

set(ORB_SLAM3_SOURCE_DIR "${PROJECT_SOURCE_DIR}/ORB-SLAM3")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${ORB_SLAM3_SOURCE_DIR}
    ${ORB_SLAM3_SOURCE_DIR}/include
    ${ORB_SLAM3_SOURCE_DIR}/include/CameraModels
    ${ORB_SLAM3_SOURCE_DIR}/Thirdparty/Sophus
    ${ORB_SLAM3_SOURCE_DIR}/Thirdparty/DBoW2
    ${ORB_SLAM3_SOURCE_DIR}/Thirdparty/g2o
    ${EIGEN3_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIR}
)

##################################################################################
##  Build the gaussian splatting component libraries
##################################################################################

add_library(simple_knn SHARED
    third_party/simple-knn/simple_knn.cu
    third_party/simple-knn/simple_knn.h
    third_party/simple-knn/spatial.cu
    third_party/simple-knn/spatial.h)
target_link_libraries(simple_knn "${TORCH_LIBRARIES}")

add_library(cuda_rasterizer SHARED
    include/operate_points.h
    src/operate_points.cu
    include/rasterize_points.h
    src/rasterize_points.cu
    include/stereo_vision.h
    src/stereo_vision.cu
    cuda_rasterizer/auxiliary.h
    cuda_rasterizer/backward.cu
    cuda_rasterizer/backward.h
    cuda_rasterizer/config.h
    cuda_rasterizer/forward.cu
    cuda_rasterizer/forward.h
    cuda_rasterizer/operate_points.h
    cuda_rasterizer/rasterizer.h
    cuda_rasterizer/rasterizer_impl.cu
    cuda_rasterizer/rasterizer_impl.h
    cuda_rasterizer/stereo_vision.h)

# 将光栅化器的架构显式修改为支持 5090 (sm_120)
set_target_properties(cuda_rasterizer PROPERTIES CUDA_ARCHITECTURES "120")
target_include_directories(cuda_rasterizer PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(cuda_rasterizer
    glm::glm
    "${TORCH_LIBRARIES}"
    Eigen3::Eigen)

##################################################################################
##  Build the ImGui library
##################################################################################

add_library(imgui SHARED
    viewer/imgui/imconfig.h
    viewer/imgui/imgui_demo.cpp
    viewer/imgui/imgui_draw.cpp
    viewer/imgui/imgui_impl_glfw.cpp
    viewer/imgui/imgui_impl_glfw.h
    viewer/imgui/imgui_impl_opengl3_loader.h
    viewer/imgui/imgui_impl_opengl3.cpp
    viewer/imgui/imgui_impl_opengl3.h
    viewer/imgui/imgui_internal.h
    viewer/imgui/imgui_tables.cpp
    viewer/imgui/imgui_widgets.cpp
    viewer/imgui/imgui.cpp
    viewer/imgui/imgui.h
    viewer/imgui/imstb_rectpack.h
    viewer/imgui/imstb_textedit.h
    viewer/imgui/imstb_truetype.h)
target_link_libraries(imgui glfw OpenGL::GL)

##################################################################################
##  Build the gaussian mapper library
##################################################################################

# 使用 GLOB 自动捕获所有 src 下的 cpp 文件，防止漏写 line 特性的实现文件
file(GLOB MAPPER_CPP_SOURCES "src/*.cpp")
add_library(gaussian_mapper SHARED
    third_party/tinyply/tinyply.h
    third_party/tinyply/tinyply.cpp
    ${MAPPER_CPP_SOURCES})

target_link_libraries(gaussian_mapper
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
    ${OpenCV_LIBRARIES}
    ${jsoncpp_LIBRARIES} # 使用修复后的 jsoncpp 变量
    "${TORCH_LIBRARIES}"
    Eigen3::Eigen
    simple_knn
    cuda_rasterizer)

##################################################################################
##  Build the viewer library
##################################################################################

add_library(gaussian_viewer SHARED
    viewer/drawer_utils.h
    viewer/imgui_viewer.cpp
    viewer/imgui_viewer.h
    viewer/imgui_viewer_line.cpp
    viewer/imgui_viewer_line.h
    viewer/map_drawer.cpp
    viewer/map_drawer.h)
target_link_libraries(gaussian_viewer
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
    gaussian_mapper
    imgui
    ${OpenCV_LIBRARIES}
    ${jsoncpp_LIBRARIES} # 使用修复后的 jsoncpp 变量
    "${TORCH_LIBRARIES}"
    glm::glm
    glfw
    OpenGL::GL)

##################################################################################
##  Build the test & mapping examples
##################################################################################

add_executable(train_colmap examples/train_colmap.cpp)
target_link_libraries(train_colmap gaussian_viewer gaussian_mapper)

add_executable(view_result examples/view_result.cpp)
target_link_libraries(view_result gaussian_viewer gaussian_mapper)

macro(add_slam_executable exec_name src_file)
    add_executable(${exec_name} ${src_file})
    target_link_libraries(${exec_name} PRIVATE gaussian_viewer gaussian_mapper ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)
endmacro()

add_slam_executable(replica_mono examples/replica_mono.cpp)
add_slam_executable(replica_rgbd examples/replica_rgbd.cpp)
add_slam_executable(tum_mono examples/tum_mono.cpp)
add_slam_executable(tum_rgbd examples/tum_rgbd.cpp)
add_slam_executable(euroc_stereo examples/euroc_stereo.cpp)

# eth3d RGBD
add_slam_executable(eth3d_rgbd examples/eth3d_rgbd.cpp)

#eth3d mono
add_slam_executable(eth3d_mono examples/eth3d_mono.cpp)

#rover RGBD
add_slam_executable(rover_rgbd examples/rover_rgbd.cpp)

#rover mono
add_slam_executable(rover_mono examples/rover_mono.cpp)



if(realsense2_FOUND)
    add_executable(realsense_rgbd examples/realsense_rgbd.cpp)
    target_include_directories(realsense_rgbd PUBLIC ${realsense_INCLUDE_DIR})
    target_link_libraries(realsense_rgbd gaussian_viewer gaussian_mapper ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so ${realsense2_LIBRARY})
endif()
