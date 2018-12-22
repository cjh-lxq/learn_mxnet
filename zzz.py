#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/20 9:39:45
# @Author  : 陈建宏
# @Site    : BeiJing
# @File    : 00_hello_world.py.py
# @Software: PyCharm Community Edition
from mxnet import nd,init
import tensorflow
import cv2
import numpy as np

if __name__ == '__main__':
    print "hello world1"
    # img=cv2.imread(r'/home/xue/test.jpg')
    # cv2.imshow('1',img)
    # cv2.waitKey(0)
    # print img.shape

    x=nd.full((2,3),5.0)
    print x
    print x.shape,x.size,x.dtype
    x=nd.ones((2,3))
    print x
    print x.shape, x.size, x.dtype
    x=nd.random.randn(2,3)
    print x
    print x.shape, x.size, x.dtype
    x=nd.random.uniform(-1,1,(2,3))
    print x
    print x.shape, x.size, x.dtype