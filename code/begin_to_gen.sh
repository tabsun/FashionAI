#!/bin/bash
export CUDA_VISIBLE_DEVICES=5
python generate_dlib_train_data.py \
--model="cmu" \
--image="../data/train/val.csv" \
--tag="dress" \
--inputsize='512' \
--resolution="368x368" \
--scales="[1.0, (0.5,0.25,1.5), (0.5,0.75,1.5), (0.25,0.5,1.5), (0.75,0.5,1.5), (0.5,0.5,1.5)]"
