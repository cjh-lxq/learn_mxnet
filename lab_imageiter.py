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

if __name__ == '__main__':
    data_iter = mx.image.ImageIter(batch_size=4, data_shape=(3, 224, 224,), label_width=1, path_imglist='train.lst')
    data_iter.reset()
    for data in data_iter:
        l=data.label[0]
        d=data.data[0]
        print d
        # print (d.shape)
