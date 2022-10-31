#!/bin/bash

#--no-cache-dir

pip list

python3 inpainting.py \
-p 'red coca cola can, realistic, 4k, 8k' \
-img img2.png -mask mask2.png \
-n 20 -gs 5 -s 0.5 -n-inf-s 200




