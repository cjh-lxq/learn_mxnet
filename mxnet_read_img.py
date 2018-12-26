#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-12-25 下午2:29
# @Author  : 陈建宏
# @Site    : BeiJing
# @File    : mxnet_read_img.py
# @Software: PyCharm Community Edition
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

PIC_SIZE = 227
BATCH_SIZE = 20
# TRAIN_PATH='/home/xue/ChenJH2018/dog_cat/train'
TRAIN_PATH = '/media/cjh/colorful/Data_set/dog_and_cat/train'


def get_pic(pic_name):
    path = TRAIN_PATH
    pic = cv2.resize(cv2.imread(os.path.join(path, pic_name)), (PIC_SIZE, PIC_SIZE))
    return pic


def get_pic_dogandcat(num_pic):
    path = TRAIN_PATH
    label_list = []
    pic_name_list = []
    pic_list = []
    for num in range(num_pic):
        label = rd.randint(0, 1)
        if label == 0:
            pic_name_list.append('cat.' + str(rd.randint(101, 12499)) + '.jpg')
            label_list.append(0)
        else:
            pic_name_list.append('dog.' + str(rd.randint(101, 12499)) + '.jpg')
            label_list.append(1)
    pool = ThreadPool(8)
    pic_list = pool.map(get_pic, pic_name_list)
    pool.close()
    pool.join()
    pic_list = np.transpose(pic_list, (0, 3, 1, 2))
    return nd.array(pic_list, dtype=np.float32), nd.array(label_list, dtype=np.float32)



if __name__ == '__main__':
    start=time.time()
    a,b=get_pic_dogandcat(10000)
    print time.time()-start