#!/usr/bin/env python


import sys
import numpy as np
import pandas as pd
import chainer
import random
from chainer.datasets import cifar


EVAL_LABEL = 'ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'


class PreprocessedImagenetDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root, mean, crop_size, random=True):
        self.base = chainer.datasets.LabeledImageDataset(path, root)
        self.mean = mean.astype('f')
        self.crop_size = crop_size
        self.random = random

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        # It reads the i-th image/label pair and return a preprocessed image.
        # It applies following preprocesses:
        #     - Cropping (random or center rectangular)
        #     - Random flip
        #     - Scaling to [0, 1] value
        crop_size = self.crop_size

        image, label = self.base[i]
        _, h, w = image.shape

        print(self.mean.shape)

        if self.random:
            # Randomly crop a region and flip the image
            top = random.randint(0, h - crop_size - 1)
            left = random.randint(0, w - crop_size - 1)
            if random.randint(0, 1):
                image = image[:, :, ::-1]
        else:
            # Crop the center
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size

        image = image[:, top:bottom, left:right]
        image -= self.mean[:, top:bottom, left:right]
        image *= (1.0 / 255.0)  # Scale to [0, 1]
        return image, label


class PreprocessedCifar100Dataset(chainer.dataset.DatasetMixin):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def get_example(self, i):
        data, label = self.data[i]
        return data, label



def get_cifar100_dataset(scale):
    train, valid = cifar.get_cifar100(scale=scale)
    train = PreprocessedCifar100Dataset(train)
    valid = PreprocessedCifar100Dataset(valid)

    return train, valid


class ImagenetDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root):
        self.data = chainer.datasets.LabeledImageDataset(path, root)

    def __len__(self):
        return len(self.data)

    def get_example(self, i):
        data, label = self.data[i]
        return data, label

