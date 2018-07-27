#!/bin/bash
export CUDA_VISIBLE_DEVICES=4

python run.py \
--modelpath="./models/trained/skirt/frozen_graph.pb" \
--imagepath="../data/new_data/test" \
--csv="../submit/submit_skirt.csv" \
--tag="skirt" \
--test="submit" \
--inputsize='368' \
--resolution="368x368" \
--scales="[1.0, (0.5,0.25,1.5), (0.5,0.75,1.5), (0.25,0.5,1.5), (0.75,0.5,1.5), (0.5,0.5,1.5)]"
