# 1. 核心底座
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 2. 系统级依赖 (保留了 unzip 和 pkg-config 等编译工具)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    pkg-config \
    python3 \
    python3-pip \
    python3-dev \
    libgl1-mesa-dev \
    libglfw3-dev \
    libglew-dev \
    libjsoncpp-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. 安装 PyTorch
RUN pip3 install --no-cache-dir --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

ENV Torch_DIR=/usr/local/lib/python3.10/dist-packages/torch/share/cmake/Torch

WORKDIR /workspace
CMD ["/bin/bash"]
