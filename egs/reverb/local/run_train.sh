#!/bin/bash
# Copyright 2018 Johns Hopkins University (Author: Aswin Shanmugam Subramanian)
# Apache 2.0

. ./path.sh

gpu_id=0
case $(hostname -f) in
  *.clsp.jhu.edu) gpu_id=`free-gpu` ;; # JHU,
esac 

if [ ! -f model/mlp.tr ]; then
    echo "Training the network"
    train.py --gpu $gpu_id
else
    echo "Not training the network. Using existing model in model/"
fi
