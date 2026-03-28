# PyTorch/CUDA stack aligned with requirements.txt (torch==2.1.1, CUDA 12.1).
# Clone the repo into WORKDIR (e.g. /workspace) at runtime; no COPY needed for build.
FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# System deps: video/OpenCV/OpenEXR-style tooling; CUDA compat for some host drivers (same pattern as common PyTorch images)
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y wget gnupg tzdata && \
    ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && apt-get install -y \
        cuda-compat-12-1 \
        git \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/usr/local/cuda-12.1/compat:$LD_LIBRARY_PATH

RUN pip install --upgrade pip

# Mirrors requirements.txt (torch/torchvision come from the base image tag, matching 2.1.1 / 0.16.1).
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121\
    numpy==1.24.0 \
    opencv-python \
    matplotlib \
    pillow \
    imageio==2.37.0 \
    imageio-ffmpeg==0.4.7 \
    decord \
    xformers==0.0.23 \
    einops==0.4.1 \
    easydict \
    tqdm \
    OpenEXR==3.3.1