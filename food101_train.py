#!/usr/bin/env python
# -*- coding: utf-8 -*-

# change https://github.com/mitmul/chainer-cifar10 to cifar100

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass

import sys

sys.path.insert(0, '/chainer/chainer')

import argparse
from functools import partial
from importlib import import_module
import json
import os
import random
import re
import shutil
import time

import numpy as np

import chainer
from chainer import iterators
from chainer import optimizers
from chainer import training
from chainer.datasets import TransformDataset
from chainer.datasets import cifar
import chainer.links as L
from chainer.training import extensions
from chainercv import transforms
import cv2 as cv
from skimage import transform as skimage_transform
from getdataset import *
import transform
import utils

from logging import getLogger
logger = getLogger('__main__')


def parse_args():
    parser = argparse.ArgumentParser(description='Training script for ImageNet'
                                     )
    parser.add_argument('--model_file', type=str, default='models/ResNet50.py')
    parser.add_argument('--model_name', type=str, default='ResNet50')
    parser.add_argument('--gpus', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train_list', type=str, default='train.txt')
    parser.add_argument('--val_list', type=str, default='val.txt')
    parser.add_argument('--train_image', type=str, default='ILSVRC2012_img_train')
    parser.add_argument('--val_image', type=str, default='ILSVRC2012_img_val')

    # Train settings
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--training_epoch', type=int, default=500)
    parser.add_argument('--initial_lr', type=float, default=0.05)
    parser.add_argument('--lr_decay_rate', type=float, default=0.5)
    parser.add_argument('--lr_decay_epoch', type=float, default=25)
    parser.add_argument('--weight_decay', type=float, default=0.0005)

    # Data augmentation settings
    parser.add_argument('--random_angle', type=float, default=15.0)
    parser.add_argument('--pca_sigma', type=float, default=25.5)
    parser.add_argument('--expand_ratio', type=float, default=1.2)
    parser.add_argument('--crop_size', type=int, nargs='*', default=[224, 224])
    parser.add_argument('--output_class', type=int, default=100)
    parser.add_argument('--mean', '-m', default='mean.npy',
                        help='Mean file (computed by compute_mean.py)')

    args = parser.parse_args()
    return args


def create_result_dir(prefix):
    result_dir = 'results/food101/{}_{}_0'.format(
        prefix, time.strftime('%Y-%m-%d_%H-%M-%S'))
    while os.path.exists(result_dir):
        i = result_dir.split('_')[-1]
        result_dir = re.sub('_[0-9]+$', result_dir, '_{}'.format(i))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    shutil.copy(__file__, os.path.join(result_dir, os.path.basename(__file__)))
    return result_dir


def main():
    args = parse_args()

    chainer.global_config.autotune = True
    #chainer.set_debug(True)

    # Set the random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Set up Devices
    devices = utils.setup_devices(args.gpus)

    # Load model
    ext = os.path.splitext(args.model_file)[1]
    model_path = '.'.join(os.path.split(args.model_file)).replace(ext, '')
    model = import_module(model_path)
    model = getattr(model, args.model_name)(args.output_class)
    #model = L.Classifier(model)
    model.to_gpu()

    # create result dir
    result_dir = create_result_dir(args.model_name)
    shutil.copy(args.model_file, os.path.join(
        result_dir, os.path.basename(args.model_file)))
    with open(os.path.join(result_dir, 'args'), 'w') as fp:
        fp.write(json.dumps(vars(args)))
    print(json.dumps(vars(args), sort_keys=True, indent=4))

    # Create Dataset
    # Load the datasets and mean file
    mean = np.load(args.mean)
    train = ImagenetDataset(args.train_list, args.train_image)
    valid = ImagenetDataset(args.val_list, args.val_image)

    train_transform = partial(
        transform.food101_transform, mean=mean, random_angle=args.random_angle,
        expand_ratio=args.expand_ratio,
        crop_size=args.crop_size, train=True)
    valid_transform = partial(transform.food101_transform, mean=mean, train=False)
    
    train = TransformDataset(train, train_transform)
    valid = TransformDataset(valid, valid_transform)

    # Create Iterator
    train_iter = chainer.iterators.MultiprocessIterator(train, args.batchsize, n_processes=4)
    val_iter = chainer.iterators.MultiprocessIterator(valid, args.batchsize, shuffle=False,
    repeat=False, n_processes=4)
    #train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    #val_iter = chainer.iterators.SerialIterator(valid, args.batchsize, repeat=False, shuffle=False)

    # Set Optimizer
    optimizer = optimizers.MomentumSGD(lr=args.initial_lr, momentum=0.9)
    optimizer.setup(model)
    if args.weight_decay > 0:
        optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))
    # optimizer.use_fp32_update()

    # Updater
    updater = training.ParallelUpdater(train_iter, optimizer, devices=devices)

    # Trainer
    trainer = training.Trainer(updater, (args.training_epoch, 'epoch'), result_dir)

    # Trainer Extensions
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.observe_lr())
    trainer.extend(extensions.Evaluator(
        val_iter, model, device=devices['main']), name='val')
    trainer.extend(extensions.ExponentialShift(
        'lr', args.lr_decay_rate), trigger=(args.lr_decay_epoch, 'epoch'))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'main/accuracy', 'val/main/loss',
         'val/main/accuracy', 'elapsed_time', 'lr']))

    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(
            ['main/loss', 'val/main/loss'], x_key='epoch', file_name='loss.png'))
        trainer.extend(extensions.PlotReport(
            ['main/accuracy', 'val/main/accuracy'], x_key='epoch',
            file_name='accuracy.png'))

    # Print progress bar
    trainer.extend(extensions.ProgressBar())

    # Save the model which minimizes validation loss
    trainer.extend(extensions.snapshot_object(model, filename='bestmodel.npz'),
                   trigger=training.triggers.MinValueTrigger('val/main/loss'))

    trainer.run()


if __name__ == '__main__':
    import logging
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    main()
