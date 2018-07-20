#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

for step in {49000..149001..2000}
do
    echo '=========================================================='
    echo $step
    python run_checkpoint.py \
--tag='skirt' \
--model='seresnet50' \
--modelpath='trained/skirt/seresnet50_batch:16_lr:0.0001_gpus:8_368x368_skirt/model-'${step}
    
    python -m tensorflow.python.tools.freeze_graph \
--input_graph=./skirt_tmp/graph.pb \
--output_graph=./skirt_tmp/frozen_graph.pb \
--input_checkpoint=./skirt_tmp/chk-1 \
--output_node_names="Openpose/concat_stage7"
    
    cp skirt_tmp/frozen_graph.pb ../models/trained/skirt/graph/

 
    python run.py \
--model="seresnet50" \
--image="../data/train/val_bak.csv" \
--tag="skirt" \
--inputsize='368' \
--resolution="368x368" \
--scales="[1.0, (0.5,0.25,1.5), (0.5,0.75,1.5), (0.25,0.5,1.5), (0.75,0.5,1.5), (0.5,0.5,1.5)]"
done

