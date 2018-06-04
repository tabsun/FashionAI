#!/bin/bash
export CUDA_VISIBLE_DEVICES=8,9
python run.py \
--model="cmu" \
--image="../data/test_b/test.csv" \
--tag="blouse" \
--test="submit" \
--resolution="368x368" \
--scales="[1.0, (0.5,0.25,1.5), (0.5,0.75,1.5), (0.25,0.5,1.5), (0.75,0.5,1.5), (0.5,0.5,1.5)]"
