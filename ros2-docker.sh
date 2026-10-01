#!/bin/bash
# ros2-docker.sh -- launch a ROS 2 Jazzy container with hardware access

# Device passthrough -- adjust if your devices differ
CAMERA_DEV="/dev/video0"
SERIAL_DEV="/dev/ttyUSB0"   # change to /dev/ttyUSB0 if that's what you have

# Only pass the serial device if it exists
DEVICE_FLAGS="--device=$CAMERA_DEV:$CAMERA_DEV"
if [ -e "$SERIAL_DEV" ]; then
    DEVICE_FLAGS="$DEVICE_FLAGS --device=$SERIAL_DEV:$SERIAL_DEV"
    echo "Passing serial device: $SERIAL_DEV"
else
    echo "No serial device at $SERIAL_DEV -- motors will be unavailable"
fi

# X11 forwarding for RViz / OpenCV windows
xhost +local:docker >/dev/null 2>&1

docker run -it --rm \
    --name medpal-ros2 \
    --privileged \
    --network=host \
    $DEVICE_FLAGS \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /home/medpal/2026medpal:/workspace \
    -w /workspace \
    medpal-ros2:full \
    bash
