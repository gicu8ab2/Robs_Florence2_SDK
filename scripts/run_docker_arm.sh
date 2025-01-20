#! /bin/bash

docker run -it \
       --runtime=nvidia \
       --gpus all \
       --privileged \
       --net=host \
       -e NVIDIA_VISIBLE_DEVICES=all \
       -e NVIDIA_DRIVER_CAPABILITIES=all \
       -v /tmp/.X11-unix:/tmp/.X11-unix \
       -v /home/ubuntu:/home/ubuntu \
       -e DISPLAY=$DISPLAY \
       florence2:r36.4.0 /bin/bash
