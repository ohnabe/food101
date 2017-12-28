#/usr/bin/env python

import numpy as np
import cv2 as cv
from chainercv import transforms


def cv_rotate(img, angle):
    img = img.transpose(1, 2, 0) / 255.0
    center = (img.shape[0] // 2, img.shape[1] // 2)
    r = cv.getRotationMatrix2D(center, angle, 1.0)
    img = cv.warpAffine(img, r, img.shape[:2])
    img = img.transpose(2, 0, 1) * 255.
    img = img.astype(np.float32)
    return img


def random_rotate(img, random_angle):
    if random_angle != 0:
        angle = np.random.uniform(-random_angle, random_angle)
        img = cv_rotate(img, angle)
    return img


def color_augmentation(img, pca_sigma):
    if pca_sigma != 0:
        img = transforms.pca_lighting(img, pca_sigma)
    return img


def standalization(img, mean, std):
    img -= mean[:, None, None]
    img /= std[:, None, None]
    return img


def substract_mean(img, mean):
    img -= mean
    return img


def random_flip(img):
    img = transforms.random_flip(img, x_random=True)
    return img


def random_expand(img, expand_ratio):
    if expand_ratio > 1:
        img = transforms.random_expand(img, max_ratio=expand_ratio)
    return img


def random_crop(img, crop_size):
    if tuple(crop_size) != (32, 32):
        img = transforms.random_crop(img, tuple(crop_size))
    return img


def resize(img, resize_size):
    img = transforms.resize(img, size=resize_size)
    return img


def mean_scale(img):
    # https://github.com/keras-team/keras/blob/master/keras/applications/imagenet_utils.py
    # def _preprocess_numpy_input(x, data_format, mode)
    img /= 127.5
    img -= 1.
    return img


def cifar_transform(inputs, mean, std, random_angle=15., pca_sigma=255., expand_ratio=1.0,
                    crop_size=(32, 32), train=True):

    img, label = inputs
    img = img.copy()

    img = random_rotate(img, random_angle)
    img = color_augmentation(img, pca_sigma)

    img = standalization(img, mean, std)

    if train:
        img = random_flip(img)
        img = random_expand(img, expand_ratio)
        img = random_crop(img, tuple(crop_size))
    return img, label


def imagenet_transform(inputs, mean, random_angle=15., expand_ratio=1.0,
                    crop_size=(224, 224), train=True):

    img, label = inputs
    img = img.copy()

    img = random_rotate(img, random_angle)

    _, h,w = mean.shape
    img = resize(img, (h, w))
    img = substract_mean(img, mean)

    if train:
        img = random_flip(img)
        img = random_expand(img, expand_ratio)
        img = random_crop(img, tuple(crop_size))
    else:
        img = resize(img, tuple(crop_size))
    return img, label


def food101_transform(inputs, mean, random_angle=15., expand_ratio=1.0,
                    crop_size=(224, 224), train=True):

    img, label = inputs
    img = img.copy()

    img = mean_scale(img)
    img = resize(img, (256, 256))

    img = random_rotate(img, random_angle)
    #_, h,w = mean.shape
    #img = resize(img, (h, w))
    #img = substract_mean(img, mean)

    if train:
        img = random_flip(img)
        img = random_expand(img, expand_ratio)
        img = random_crop(img, tuple(crop_size))
    else:
        img = resize(img, tuple(crop_size))
    return img, label
