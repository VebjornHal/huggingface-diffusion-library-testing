#!/bin/bash

#--no-cache-dir

python3 inpainting.py \
-p 'treehouse, top of a big tree, in the middle of a field of green grass' \
-img img4.png -mask mask4_1.png \
-n 10 -gs 7.5 -s 0.8 -n-inf-s 500
