#!/bin/bash

python dump_model_params.py \
--meta ./models/numpy/seresnet101/se_resnet101.ckpt.meta \
./models/numpy/seresnet101/se_resnet101.ckpt \
./models/numpy/se_resnet101.npy
