#!/bin/bash
# cmu or vgg
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,8,7
python train.py \
--model 'seresnet50' \
--datapath '../data/' \
--imgpath '../data/' \
--batchsize 16 \
--gpus 8 \
--max-epoch 23 \
--lr '0.0001' \
--pretrain_basepath './models/' \
--pretrain_path 'numpy/se_resnet50.npy' \
--modelpath './models/training' \
--logpath './models/training/' \
--checkpoint '' \
--tag 'skirt' \
--input-width 368 \
--input-height 368


