FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ENV COLMAP_VERSION=3.7
ENV CMAKE_VERSION=3.21.0
ENV OPENCV_VERSION=4.5.5.62

RUN echo "Installing apt packages..." \
	&& export DEBIAN_FRONTEND=noninteractive \
	&& apt -y update --no-install-recommends \
    && apt install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
	&& apt -y install --no-install-recommends \
	git \
	cmake \
	wget \
	ffmpeg \
	tk-dev \
	libxi-dev \
	libc6-dev \
	libbz2-dev \
	libffi-dev \
	libomp-dev \
	libssl-dev \
	zlib1g-dev \
	libcgal-dev \
	libgdbm-dev \
	libglew-dev \
	python3.9-dev \
	qtbase5-dev \
	checkinstall \
	libglfw3-dev \
	libeigen3-dev \
	libgflags-dev \
	libxrandr-dev \
	libopenexr-dev \
	libsqlite3-dev \
	libxcursor-dev \
	build-essential \
	libcgal-qt5-dev \
	libxinerama-dev \
	libboost-all-dev \
	libfreeimage-dev \
	libncursesw5-dev \
	libatlas-base-dev \
	libqt5opengl5-dev \
	libgoogle-glog-dev \
	libsuitesparse-dev \
	python3.9-distutils \
    python3-pip \
	&& apt autoremove -y \
	&& apt clean -y \
	&& export DEBIAN_FRONTEND=dialog \
    && alias python=python3.9

RUN python3.9 -m pip install --no-cache-dir torch torchvision torchaudio
