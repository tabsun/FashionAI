#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python run_checkpoint.py \
--model=seresnext50 \
--tag='trousers' \
--modelpath='trained/trousers/model-94000'

python -m tensorflow.python.tools.freeze_graph \
--input_graph=./tmp/graph.pb \
--output_graph=./tmp/frozen_graph.pb \
--input_checkpoint=./tmp/chk-1 \
--output_node_names="Openpose/concat_stage7"

rm ./tmp/chk-1* -f
rm ./tmp/checkpoint
rm ./tmp/graph.pb
