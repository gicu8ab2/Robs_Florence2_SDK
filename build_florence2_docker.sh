#! /bin/bash

docker buildx build --platform linux/arm64 -f Dockerfile_r36.4.0 -t florence2:r36.4.0
