#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/26 9:59:55
# @Author  : 陈建宏
# @Site    : BeiJing
# @File    : mxnet_vgg_gpu.py
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
TRAIN_DATA_SIZE = 500
TEST_DATA_SIZE = 300
DROPOUT_RATE = 0.5


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


def accuracy(y_hat, y):
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()


def evaluate_accuracy(data_iter, net):
    acc = 0
    for X, y in data_iter:
        X=X.copyto(mx.gpu())
        y=y.copyto(mx.gpu())
        acc += accuracy(net(X), y)
    return acc / len(data_iter)


def vgg():
    print 'programe begin'
    train_pic_list, train_label_list = get_pic_dogandcat(TRAIN_DATA_SIZE)
    test_pic_list, test_label_list = get_pic_dogandcat(TEST_DATA_SIZE)
    train_dataset = gluon.data.ArrayDataset(train_pic_list, train_label_list)
    train_data_iter = gluon.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = gluon.data.ArrayDataset(test_pic_list, test_label_list)
    test_data_iter = gluon.data.DataLoader(test_dataset, batch_size=TEST_DATA_SIZE, shuffle=True)
    # train_data_iter = mx.image.ImageIter(batch_size=BATCH_SIZE, data_shape=(3, 224, 224),
    #                                      path_imglist='train.lst')
    # train_data_iter.reset()
    print 'dataset created'
    with mx.Context(mx.gpu()):
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
        print 'net created'
        net.initialize()
        ####################
        X = nd.random.uniform(shape=(BATCH_SIZE, 3, 224, 224))
        for blk in net:
            X = blk(X)
            print(blk.name, 'output shape:\t', X.shape)
        # exit()
        # net(train_pic_list[0])
        ####################
        loss = gluon.loss.SoftmaxCrossEntropyLoss()
        learning_rate = 0.01
        trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': learning_rate})
        num_epoch = 1000
        print 'train begin'
        for epoch in range(num_epoch):
            start_time = time.time()
            train_l_sum = 0
            train_acc_sum = 0
            num_add = 1
            for X,y in train_data_iter:
                X=X.copyto(mx.gpu())
                y=y.copyto(mx.gpu())
                with autograd.record():
                    y_predict = net(X)
                    l = loss(net(X), y)
                l.backward()
                trainer.step(batch_size=BATCH_SIZE)
                train_l_sum += l.mean().asscalar()
                train_acc_sum += gb.accuracy(y_predict, y)
                num_add += 1
            if (epoch != 0 and epoch % 3 == 0):
                learning_rate = learning_rate * 0.9
                trainer.set_learning_rate(learning_rate)
            test_acc = evaluate_accuracy(test_data_iter, net)
            # test_acc = 0
            time1 = time.time() - start_time
            print('epoch %3d, loss %.8f, train acc %.8f, test acc %.8f, learning_rate: %.8f, Time: %.3fs, predict_time: %dmin%ds'
                    % (epoch + 1, train_l_sum / num_add,
                       train_acc_sum / num_add, test_acc, trainer.learning_rate,
                       time1, int(time1 * (num_epoch - epoch - 1) / 60), time1 * (num_epoch - epoch - 1) % 60))


if __name__ == '__main__':
    # get_pic('dog.5488.jpg')
    # img=mx.image.imread('/media/cjh/DATA/dxr/dog_and_cat/train/dog.5488.jpg')
    # print img.shape,img.dtype
    # img2=nd.array(cv2.imread('/media/cjh/DATA/dxr/dog_and_cat/train/dog.5488.jpg'))
    # print img2.shape,img2.dtype
    #
    # img = img.reshape((1, 3, 224, 224))
    # print img.shape, img.dtype
    vgg()
