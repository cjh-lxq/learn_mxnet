#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/26 15:45:53
# @Author  : 陈建宏
# @Site    : BeiJing
# @File    : mnist vgg.py
# @Software: PyCharm
from __future__ import division
from mxnet import nd, init, autograd
from mxnet import gluon
from mxnet.gluon import nn
import numpy as np
import os, cv2, time
import random as rd
import gluonbook as gb
import mxnet as mx
from mxnet.gluon import loss as gloss
from gluonbook import accuracy,evaluate_accuracy
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

PIC_SIZE = 224
BATCH_SIZE = 20
# TRAIN_PATH='/home/xue/ChenJH2018/dog_cat/train'
TRAIN_PATH = '/media/cjh/colorful/Data_set/dog_and_cat/train'
TEST_PATH = '/media/cjh/colorful/Data_set/dog_and_cat/test2'
TRAIN_REC_PATH = '/home/cjh/lab_pycharm/data/train.rec'
TRAIN_DATA_SIZE = 300
TEST_DATA_SIZE = 300
DROPOUT_RATE = 0.5


def get_iter():
    train_data_iter = mx.image.ImageIter(batch_size=BATCH_SIZE, data_shape=(3, PIC_SIZE, PIC_SIZE),
                                         path_imglist='train.lst')
    test_data_iter = mx.image.ImageIter(batch_size=BATCH_SIZE, data_shape=(3, PIC_SIZE, PIC_SIZE),
                                        path_imglist='test.lst')
    return train_data_iter,test_data_iter

def vgg():
    net = nn.Sequential()
    net.add(
        nn.Conv2D(channels=64, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), activation='relu', in_channels=3))
    net.add(
        nn.Conv2D(channels=64, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), activation='relu', in_channels=64))
    net.add(nn.MaxPool2D(pool_size=2, strides=2))
    net.add(
        nn.Conv2D(channels=128, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), activation='relu', in_channels=64))
    net.add(
        nn.Conv2D(channels=128, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), activation='relu', in_channels=128))
    net.add(nn.MaxPool2D(pool_size=2, strides=2))
    net.add(
        nn.Conv2D(channels=256, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), activation='relu', in_channels=128))
    net.add(
        nn.Conv2D(channels=256, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), activation='relu', in_channels=256))
    net.add(
        nn.Conv2D(channels=256, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), activation='relu', in_channels=256))
    net.add(nn.MaxPool2D(pool_size=2, strides=2))
    net.add(
        nn.Conv2D(channels=512, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), activation='relu', in_channels=256))
    net.add(
        nn.Conv2D(channels=512, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), activation='relu', in_channels=512))
    net.add(
        nn.Conv2D(channels=512, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), activation='relu', in_channels=512))
    net.add(nn.MaxPool2D(pool_size=2, strides=2))
    net.add(
        nn.Conv2D(channels=512, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), activation='relu', in_channels=512))
    net.add(
        nn.Conv2D(channels=512, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), activation='relu', in_channels=512))
    net.add(
        nn.Conv2D(channels=512, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), activation='relu', in_channels=512))
    net.add(nn.MaxPool2D(pool_size=2, strides=2))
    net.add(nn.Dense(4096, activation='relu', in_units=7 * 7 * 512))
    net.add(nn.Dropout(DROPOUT_RATE))
    net.add(nn.Dense(4096, activation='relu', in_units=4096))
    net.add(nn.Dropout(DROPOUT_RATE))
    net.add(nn.Dense(2, in_units=4096))
    return net


def train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx,
              num_epochs):
    """Train and evaluate a model on CPU or GPU."""
    print('training on', ctx)
    loss = gloss.SoftmaxCrossEntropyLoss()
    for epoch in range(1, num_epochs + 1):
        train_l_sum = 0
        train_acc_sum = 0
        num_add=0
        start = time.time()
        for X, y in train_iter:
            X, y = X.as_in_context(ctx), y.as_in_context(ctx)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            trainer.step(batch_size)
            train_l_sum += l.mean().asscalar()
            train_acc_sum += accuracy(y_hat, y)
        test_acc = evaluate_accuracy(test_iter, net, ctx)
        num_add+=1
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '
              'time %.1f sec'
              % (epoch, train_l_sum / num_add,
                 train_acc_sum / num_add, test_acc, time.time() - start))


if __name__ == '__main__':
    # conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    # ratio = 4
    # small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
    # print small_conv_arch
    net = vgg()
    net.initialize(init.Xavier())
    X = nd.random.uniform(shape=(1, 3, 224, 224))
    for blk in net:
        X = blk(X)
        print(blk.name, 'output shape:\t', X.shape)
