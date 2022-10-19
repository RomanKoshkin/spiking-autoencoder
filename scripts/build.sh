#!/bin/bash

echo "working directory: `pwd`"

g++ \
-std=gnu++14 \
-march=native \
-ftree-vectorize \
-Ofast \
-shared \
-fPIC \
-mavx \
src/bmm_10c_small.cpp \
-o modules/bmm.dylib

echo "c++ module built in: modules/"