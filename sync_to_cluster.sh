#!/bin/bash

# Script for syncing necessary files
rsync -a -P  cluster_dir/ springfield:~/diffusion_lib_test

# Testing new command
#rsync -a -P --timeout=600 -r -zp --block-size=131072 cluster_dir/ springfield:~/diffusion_lib_test



