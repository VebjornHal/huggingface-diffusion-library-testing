#!/bin/bash

# Script for syncing necessary files
rsync -a -P springfield:~/diffusion_lib_test/ cluster_dir
