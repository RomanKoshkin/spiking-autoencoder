#!/bin/bash

echo "working directory: `pwd`"

cd ../assets && \
ffmpeg \
-pattern_type glob \
-y \
-r 60 \
-i "*.png" \
-vcodec libx264 \
-pix_fmt yuv420p \
-hide_banner \
-loglevel error \
output.mp4

echo "video is ready in `pwd`"