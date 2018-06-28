#!/bin/bash

python dump_model_params.py \
--meta ./SENET/model/se_resnet50.ckpt.meta \
./SENET/model/se_resnet50.ckpt \
se_resnet50.npy
