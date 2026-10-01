FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive

# --- ROS 2 packages ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-navigation2 \
        ros-jazzy-nav2-bringup \
        ros-jazzy-slam-toolbox \
        ros-jazzy-depthimage-to-laserscan \
        ros-jazzy-cv-bridge \
        ros-jazzy-image-transport \
        ros-jazzy-tf2-tools \
        ros-jazzy-rviz2 \
        ros-jazzy-rqt-graph \
    && rm -rf /var/lib/apt/lists/*

# --- Build and system deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git pkg-config \
        pybind11-dev python3-dev python3-pybind11 \
        libusb-1.0-0-dev libudev-dev \
        libopencv-dev python3-opencv \
        python3-pip v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# --- Python serial ---
RUN pip install --break-system-packages --no-cache-dir pyserial

# --- Build pyorbbecsdk at the exact commit ---
WORKDIR /opt/sdk
RUN git clone https://github.com/orbbec/pyorbbecsdk.git && \
    cd pyorbbecsdk && \
    git checkout ee32b47 && \
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())") && \
    make -j4

# --- Environment for the runtime container ---
ENV PYTHONPATH=/opt/sdk/pyorbbecsdk/build:$PYTHONPATH
RUN echo 'source /opt/ros/jazzy/setup.bash' >> /root/.bashrc

WORKDIR /workspace
CMD ["bash"]
