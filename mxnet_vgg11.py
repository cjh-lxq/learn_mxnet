#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/27 14:48:54
# @Author  : 陈建宏
# @Site    : BeiJing
# @File    : mxnet_vgg11.py
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

PIC_SIZE = 224
BATCH_SIZE = 20
# TRAIN_PATH='/home/xue/ChenJH2018/dog_cat/train'
TRAIN_PATH = '/media/cjh/colorful/Data_set/dog_and_cat/train'
TEST_PATH = '/media/cjh/colorful/Data_set/dog_and_cat/test2'
TRAIN_REC_PATH = '/home/cjh/lab_pycharm/data/train.rec'
TRAIN_DATA_SIZE = 300
TEST_DATA_SIZE = 300
DROPOUT_RATE = 0.5
LEARNING_RATE = 0.01


def creat_iter():
    train_data_iter = mx.image.ImageIter(batch_size=BATCH_SIZE, data_shape=(3, 224, 224), label_width=1,
                                         path_imglist='train.lst')
    train_data_iter.reset()
    test_data_iter = mx.image.ImageIter(batch_size=BATCH_SIZE, data_shape=(3, 224, 224), label_width=1,
                                        path_imglist='test.lst')
    test_data_iter.reset()
    return train_data_iter, test_data_iter


def vgg11():
    net = nn.Sequential()
    net.add(nn.Conv2D(channels=64, kernel_size=3, padding=1, activation='relu'))
    net.add(nn.MaxPool2D(pool_size=2, strides=2))
    net.add(nn.Conv2D(channels=128, kernel_size=3, padding=1, activation='relu'))
    net.add(nn.MaxPool2D(pool_size=2, strides=2))
    net.add(nn.Conv2D(channels=256, kernel_size=3, padding=1, activation='relu'))
    net.add(nn.Conv2D(channels=256, kernel_size=3, padding=1, activation='relu'))
    net.add(nn.MaxPool2D(pool_size=2, strides=2))
    net.add(nn.Conv2D(channels=512, kernel_size=3, padding=1, activation='relu'))
    net.add(nn.Conv2D(channels=512, kernel_size=3, padding=1, activation='relu'))
    net.add(nn.MaxPool2D(pool_size=2, strides=2))
    net.add(nn.Conv2D(channels=512, kernel_size=3, padding=1, activation='relu'))
    net.add(nn.Conv2D(channels=512, kernel_size=3, padding=1, activation='relu'))
    net.add(nn.MaxPool2D(pool_size=2, strides=2))
    net.add(nn.Dense(1000, activation='relu'))
    net.add(nn.Dropout(DROPOUT_RATE))
    net.add(nn.Dense(1000, activation='relu'))
    net.add(nn.Dropout(DROPOUT_RATE))
    net.add(nn.Dense(2))
    net.initialize(init.Xavier(), ctx=mx.gpu())
    return net


if __name__ == '__main__':
    train_iter, test_iter = creat_iter()
    net = vgg11()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': LEARNING_RATE})
    num_epochs = 1000
    for epoch in range(num_epochs):
        total_time_start=time.time()
        train_iter.reset()
        test_iter.reset()
        num_batch = 1
        train_l_sum = 0
        train_acc_sum = 0
        train_time_sum = 0
        read_time_sum = 0
        test_time_sum = 0
        for data_cpu in train_iter:
            start1 = time.time()
            X = data_cpu.data[0].copyto(mx.gpu())/255
            Y_true = data_cpu.label[0].copyto(mx.gpu())
            read_time_sum += (time.time() - start1)
            start2 = time.time()
            with autograd.record():
                y_predict = net(X)
                l = loss(y_predict, Y_true)
            l.backward()
            trainer.step(batch_size=BATCH_SIZE)
            train_time_sum += (time.time() - start2)
            start3 = time.time()
            train_l_sum += l.mean().asscalar()
            train_acc_sum += gb.accuracy(y_predict, Y_true)
            num_batch += 1
            test_time_sum += (time.time() - start3)
            print epoch, num_batch, train_l_sum / num_batch, train_acc_sum / num_batch
            print 'read_time_sum:', read_time_sum, '  arg:', read_time_sum / num_batch
            print 'train_time_sum:', train_time_sum, '  arg:', train_time_sum / num_batch
            print 'test_time_sum:', test_time_sum, '  arg:', test_time_sum / num_batch
            print 'total_time:',total_time_start-time.time()
