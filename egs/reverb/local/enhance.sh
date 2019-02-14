#!/bin/bash
# Copyright 2018 Johns Hopkins University (Author: Aswin Shanmugam Subramanian)
# Apache 2.0

. ./path.sh

gpu_id=0
case $(hostname -f) in
  *.clsp.jhu.edu) gpu_id=`free-gpu` ;; # JHU,
esac 

local/enhance.py data/dt_real_input_files data/dt_real_output_files_dnnwpe --gpu $gpu_id -r 1
local/enhance.py data/dt_simu_input_files data/dt_simu_output_files_dnnwpe --gpu $gpu_id
local/enhance.py data/dt_real_input_files data/dt_real_output_files_dnn --gpu $gpu_id -s 1
local/enhance.py data/dt_simu_input_files data/dt_simu_output_files_dnn --gpu $gpu_id -s 1
local/enhance.py data/et_real_input_files data/et_real_output_files_dnnwpe --gpu $gpu_id -r 1
local/enhance.py data/et_simu_input_files data/et_simu_output_files_dnnwpe --gpu $gpu_id
local/enhance.py data/et_real_input_files data/et_real_output_files_dnn --gpu $gpu_id -s 1
local/enhance.py data/et_simu_input_files data/et_simu_output_files_dnn --gpu $gpu_id -s 1
