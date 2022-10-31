#!/bin/bash

#--no-cache-dir

#pip3 install  -r requirements.txt

python3 text2img.py \
-p 'x-ray of brain with tumor, 4k, 8k' \
-n 10 -gs 7.5 -H 800 -W 800  -n-inf-s 300