FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ENV CERES_SOLVER_VERSION=2.1.0
ENV COLMAP_VERSION=3.7
ENV CMAKE_VERSION=3.21.0
ENV OPENCV_VERSION=4.5.5.62

# Specify your CUDA architecture, separated by ";"
ARG CUDA_ARCHITECTURES=80
ARG USER_ID=1000

# Create non root user and setup environment.
RUN useradd -m -d /home/user -g root -G sudo -u ${USER_ID} user
RUN usermod -aG sudo user
# Set user password
RUN echo "user:user" | chpasswd
# Ensure sudo group users are not asked for a password when using sudo command by ammending sudoers file
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Switch to new uer and workdir.
USER ${USER_ID}

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
    libboost-all-dev \
    libgtk-3-dev \
    libavdevice-dev \
    libavcodec-dev \
    libgl1-mesa-glx \
    libxxf86vm-dev \
    libembree-dev \
    libassimp-dev \
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
    libopencv-dev \
    doxygen \
    graphviz \
	python3.9-distutils \
    python3-pip \
	&& apt autoremove -y \
	&& apt clean -y \
	&& export DEBIAN_FRONTEND=dialog \
    && alias python=python3.9

RUN python3.9 -m pip install --no-cache-dir torch torchvision torchaudio

RUN echo "Installing Ceres Solver ver. ${CERES_SOLVER_VERSION}..." \
	&& cd /opt \
	&& git clone https://github.com/ceres-solver/ceres-solver \
	&& cd ./ceres-solver \
	&& git checkout ${CERES_SOLVER_VERSION} \
	&& mkdir ./build \
	&& cd ./build \
	&& cmake ../ -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF \
	&& make -j \
	&& make install \
    && rm -r build

RUN echo "Installing COLMAP ver. ${COLMAP_VERSION}..." \
	&& cd /opt \
	&& git clone https://github.com/colmap/colmap \
	&& cd ./colmap \
	&& git checkout ${COLMAP_VERSION} \
	&& mkdir ./build \
	&& cd ./build \
	&& cmake -DCUDA_NVCC_FLAGS="--std c++14" ../ \
	&& make -j \
	&& make install \
	&& colmap -h \
    && rm -r build
