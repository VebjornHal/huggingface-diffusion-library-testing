#!/bin/bash

#--no-cache-dir

pip list

python3 inpainting_strength_testing.py \
#-p 'red coca cola can, realistic, 4k, 8k' \
#-img img2.png -mask mask2.png \
#-H 600 -W 600 \
#-n 5 -gs 5 -s 0.5 -n-inf-s 200




