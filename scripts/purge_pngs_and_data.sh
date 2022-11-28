#!/bin/bash

cd ../assets && ls | grep ".png" | xargs rm && \
cd ../tmp && ls | grep ".p" | xargs rm