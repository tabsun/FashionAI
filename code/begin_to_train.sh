#!/bin/bash
# cmu or vgg
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python train.py \
--model 'vgg' \
--datapath '../data/' \
--imgpath '../data/' \
--batchsize 96 \
--gpus 8 \
--max-epoch 40 \
--lr '0.0005' \
--pretrain_basepath './models/' \
--modelpath './models/trained/blouse/' \
--logpath './models/trained/blouse/' \
--checkpoint '' \
--tag 'blouse' \
--input-width 368 \
--input-height 368

