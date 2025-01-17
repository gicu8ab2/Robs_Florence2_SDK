#! /bin/bash

docker run -it --gpus all --privileged \
       --net=host \
       -v /tmp/.X11-unix:/tmp/.X11-unix \
       -v /home/ubuntu/robert.taylor:/home/ubuntu/robert.taylor \
       -e DISPLAY=$DISPLAY \
       282244745782.dkr.ecr.us-east-1.amazonaws.com/py/src/clarifai/research:florence2 /bin/bash
