#! /bin/bash

docker buildx build --platform linux/arm64 -f Dockerfile_arm64 -t florence2:arm64 .
