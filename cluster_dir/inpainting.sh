#!/bin/bash

#--no-cache-dir

python3 inpainting.py \
-p 'two zebras running in a field, 4k, 8k' \
-img img4.png -mask mask4.png
-n 10 -gs 10 -s 0.5 -n-inf-s 500