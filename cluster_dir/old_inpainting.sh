#!/bin/bash

#--no-cache-dir

pip list

python3 old_inpainting.py \
-p 'red cola can on table, realistic' \
-img img2.png -mask mask2.png \
-n 20 -gs 20 -s 0.6 -n-inf-s 500


