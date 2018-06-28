#!/bin/bash
# cmu or vgg
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,8,7
python train.py \
--model 'seresnet50' \
--datapath '/home/shy/projects/tf-openpose/data/' \
--imgpath '/home/shy/projects/tf-openpose/data/' \
--batchsize 16 \
--gpus 8 \
--max-epoch 23 \
--lr '0.0001' \
--pretrain_basepath '/home/shy/projects/tf-openpose/models/' \
--pretrain_path 'numpy/se_resnet50.npy' \
--modelpath '/home/shy/projects/tf-openpose/models/trained/trousers' \
--logpath '/home/shy/projects/tf-openpose/models/trained/trousers/' \
--checkpoint '' \
--tag 'trousers' \
--input-width 368 \
--input-height 368


