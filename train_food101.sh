#!/usr/bin/env bash

nvidia-docker run --rm -it -v $(pwd):/work -v /srv/data/food101/food-101:/data -w /work actcast_cl:chainer3.2.0_git \
python3 /work/food101_train.py \
--model_file models/mobilenet.py \
--model_name MobileNet \
--batchsize 64 \
--training_epoch 500 \
--initial_lr 0.01 \
--lr_decay_rate 0.5 \
--lr_decay_epoch 70 \
--weight_decay 0.00000005 \
--train_image /data/images \
--val_image /data/images \
--train_list /data/meta/annotation_train_rgb.txt \
--val_list /data/meta/annotation_test_rgb.txt \
--random_angle 15.0 \
--expand_ratio 1.2 \
--crop_size 224 224 \
--seed 0 \
--gpus main=0 \
--mean mean.npy \
--output_class 101
