#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python run.py \
--modelpath="./tmp/frozen_graph.pb" \
--imagepath="../data/train" \
--csv="../data/train/val.csv" \
--tag="skirt" \
--inputsize='368' \
--resolution="368x368" \
--scales="[1.0, (0.5,0.25,1.5), (0.5,0.75,1.5), (0.25,0.5,1.5), (0.75,0.5,1.5), (0.5,0.5,1.5)]"
