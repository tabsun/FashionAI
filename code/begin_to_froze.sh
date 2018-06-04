#!/bin/bash

python run_checkpoint.py \
--model=cmu \
--modelpath='trained/blouse/vgg_batch:96_lr:0.0005_gpus:8_368x368_blouse/model-50721'

python -m tensorflow.python.tools.freeze_graph \
--input_graph=./tmp/graph.pb \
--output_graph=./tmp/frozen_graph.pb \
--input_checkpoint=./tmp/chk-1 \
--output_node_names="Openpose/concat_stage7"

rm ./tmp/chk-1* -f
rm ./tmp/checkpoint
rm ./tmp/graph.pb
