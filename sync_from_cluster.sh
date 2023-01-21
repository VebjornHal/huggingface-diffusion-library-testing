#!/bin/bash

# Script for syncing necessary files
rsync -a -P springfield:~/diffusion_lib_test/ cluster_dir

# Testing new command
#rsync -a -P --timeout=600 -r -zp --block-size=131072 springfield:~/diffusion_lib_test/ cluster_dir


