#!/bin/bash

# DATADIR="/home/mcarilli/Desktop/pt18data/apex_stale/examples/imagenet/bare_metal_train_val/"
# DATADIR="/opt/home/apex/examples/imagenet/"
cp ../common/* .
# bash run_test.sh single_gpu $1
bash run_test_npu.sh single_npu $1 $2
