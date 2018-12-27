#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/26 9:26:22
# @Author  : 陈建宏
# @Site    : BeiJing
# @File    : lab_imageiter.py
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
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool


def creat_iter():
    data_iter = mx.image.ImageIter(batch_size=20, data_shape=(3, 224, 224), label_width=1, path_imglist='train.lst')
    data_iter.reset()
    return data_iter


def smallnet():
    net = nn.Sequential()
    net.add(nn.Dense(1000, activation='relu'))
    net.add(nn.Dense(1000, activation='relu'))
    net.add(nn.Dense(1000, activation='relu'))
    net.add(nn.Dense(1000, activation='relu'))
    net.initialize(init.Xavier(), ctx=mx.gpu())
    return net


if __name__ == '__main__':
    num_epochs = 1000
    batch_size = 20
    data_iter = creat_iter()
    net = smallnet()
    # X = nd.random.uniform(shape=(batch_size, 3, 224, 224),ctx=mx.gpu())
    for epochs in range(num_epochs):
        num_batch = 1
        for data_cpu in data_iter:
            X = data_cpu.data[0].copyto(mx.gpu())
            Y = data_cpu.label[0].copyto(mx.gpu())
            out = net(X)
            print 'batch:', num_batch
            num_batch += 1
        data_iter.reset()  # 每个epoch一定要重置data_iter，非常重要
        print 'epoch:', epochs
