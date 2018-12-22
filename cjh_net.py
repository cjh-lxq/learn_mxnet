#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/22 10:04:51
# @Author  : 陈建宏
# @Site    : BeiJing
# @File    : cjh_net.py
# @Software: PyCharm
from __future__ import division
from mxnet import nd,init,autograd
from mxnet import gluon
import numpy as np
import os,cv2,time
import random as rd
import gluonbook as gb
import mxnet as mx
PIC_SIZE=227
BATCH_SIZE=20
TRAIN_PATH='/home/xue/ChenJH2018/dog_cat/train'
def pic3to1(img):
    b,g,r=cv2.split(img)
    len_pic=img.shape[0]*img.shape[1]
    b_arr = np.array(b).reshape(len_pic)
    g_arr = np.array(g).reshape(len_pic)
    r_arr = np.array(r).reshape(len_pic)
    img_arr=np.concatenate((b_arr,g_arr,r_arr))
    return img_arr
def get_pic_dogandcat(num_pic):
    path = TRAIN_PATH
    label_list=[]
    pic_list=[]
    for num in range(num_pic):
        label=rd.randint(0,1)
        if label==0:
            pic_name = 'cat.' + str(rd.randint(101, 12499)) + '.jpg'
            pic = cv2.imread(os.path.join(path, pic_name))
            pic = cv2.resize(pic, (PIC_SIZE, PIC_SIZE))/255
            pic=pic3to1(pic)
            pic_list.append(pic)
            label_list.append(0)
        else:
            pic_name = 'dog.' + str(rd.randint(101, 12499)) + '.jpg'
            pic = cv2.imread(os.path.join(path, pic_name))
            pic = cv2.resize(pic, (PIC_SIZE, PIC_SIZE))/255
            pic=pic3to1(pic)
            pic_list.append(pic)
            label_list.append(1)
        # print pic_name
    return nd.array(pic_list,dtype=np.float32),nd.array(label_list,dtype=np.float32)
def accuracy(y_hat, y):
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()
def evaluate_accuracy(data_iter, net):
    acc = 0
    for X, y in data_iter:
        acc += accuracy(net(X), y)
    return acc / len(data_iter)
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, trainer=None):
    for epoch in range(num_epochs):
        start_time=time.time()
        train_l_sum = 0
        train_acc_sum = 0
        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            if trainer is None:
                gb.sgd(params, lr, batch_size)
            else:
                trainer.step(batch_size)  # 下一节将用到。
            train_l_sum += l.mean().asscalar()
            train_acc_sum += accuracy(y_hat, y)
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, Time: %.3f'
              % (epoch + 1, train_l_sum / len(train_iter),
                 train_acc_sum / len(train_iter), test_acc, time.time()-start_time))
if __name__ == '__main__':
    print "train_start"
    with mx.Context(mx.gpu()):
        train_pic_list, train_label_list=get_pic_dogandcat(2000)
        test_pic_list,test_label_list=get_pic_dogandcat(100)
        print 'dataset is created'
        train_dataset=gluon.data.ArrayDataset(train_pic_list, train_label_list)
        train_data_iter=gluon.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_dataset=gluon.data.ArrayDataset(test_pic_list,test_label_list)
        test_data_iter=gluon.data.DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True)
        net=gluon.nn.Sequential()
        net.add(gluon.nn.Dense(2000, activation='relu'))
        net.add(gluon.nn.Dropout(0.5))
        net.add(gluon.nn.Dense(2000, activation='relu'))
        net.add(gluon.nn.Dropout(0.5))
        net.add(gluon.nn.Dense(2000, activation='relu'))
        net.add(gluon.nn.Dropout(0.5))
        net.add(gluon.nn.Dense(2))
        net.initialize(init.Normal(sigma=0.03))
        loss=gluon.loss.SoftmaxCrossEntropyLoss()
        trainer=gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.001})
        num_eproch=100
        for epoch in range(num_eproch):
            start_time=time.time()
            train_l_sum = 0
            train_acc_sum = 0
            num_add=0
            for X,y in train_data_iter:
                with autograd.record():
                    y_predict=net(X)
                    l=loss(net(X),y)
                l.backward()
                trainer.step(batch_size=BATCH_SIZE)
                train_l_sum += l.mean().asscalar()
                train_acc_sum += gb.accuracy(y_predict, y)
                num_add+=1
            test_acc = evaluate_accuracy(test_data_iter, net)
            time1=time.time() - start_time
            print('epoch %3d, loss %.8f, train acc %.8f, test acc %.8f, Time: %.3fs, predict_time: %dmin%ds'
                  % (epoch + 1, train_l_sum / len(train_data_iter),
                     train_acc_sum / len(train_data_iter), test_acc, time1,int(time1*(num_eproch-epoch-1)/60),time1*(num_eproch-epoch-1)%60))