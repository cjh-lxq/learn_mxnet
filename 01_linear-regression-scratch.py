#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/20 9:39:45
# @Author  : 陈建宏
# @Site    : BeiJing
# @File    : 00_hello_world.py.py
# @Software: PyCharm Community Edition
from __future__ import  division
from __future__ import print_function
from matplotlib import pyplot as plt
from mxnet import nd,autograd
import mxnet as mx
import random
# 在训练模型的时候，我们需要遍历数据集并不断读取小批量数据样本。
# 这里我们定义一个函数：它每次返回batch_size（批量大小）个随机样本的特征和标签。
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的。
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i: min(i + batch_size, num_examples)])
        yield features.take(j), labels.take(j)  # take 函数根据索引返回对应元素。

def linreg(X, w, b):  # 本函数已保存在 gluonbook 包中方便以后使用。
    return nd.dot(X, w) + b
def squared_loss(y_hat, y):  # 本函数已保存在 gluonbook 包中方便以后使用。
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
def sgd(params, lr, batch_size):  # 本函数已保存在 gluonbook 包中方便以后使用。
    for param in params:
        param[:] = param - lr * param.grad / batch_size
if __name__ == '__main__':
    gpu_device = mx.gpu()
    with mx.Context(gpu_device):
        debug_flage=True
        # create_database
        num_inputs = 2
        num_examples = 1000
        true_w = [2, -3.4]
        true_b = 4.2
        features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
        labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
        labels += nd.random.normal(scale=0.01, shape=labels.shape)
        if debug_flage:
            print(features[1], labels[1])
            # plt.scatter(features[:, 1].asnumpy(), labels.asnumpy(), 1)
            # plt.show()
            for X, y in data_iter(10, features, labels):
                print('train_data_iter:')
                print(X, y)
                break
        # 我们将权重初始化成均值为 0 标准差为 0.01 的正态随机数，偏差则初始化成 0。
        w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
        b = nd.zeros(shape=(1,))
        # 之后的模型训练中，我们需要对这些参数求梯度来迭代参数的值，因此我们需要创建它们的梯度
        w.attach_grad()
        b.attach_grad()
        # 在训练中，我们将多次迭代模型参数。在每次迭代中，我们根据当前读取的小批量数据样本（特征X和标签y），
        # 通过调用反向函数backward计算小批量随机梯度，并调用优化算法sgd迭代模型参数。由于我们之前设批量大小batch_size为 10，
        # 每个小批量的损失l的形状为（10，1）。回忆一下“自动求梯度”一节。由于变量l并不是一个标量，
        # 运行l.backward()将对l中元素求和得到新的变量，再求该变量有关模型参数的梯度。

        # 在一个迭代周期（epoch）中，我们将完整遍历一遍data_iter函数，并对训练数据集中所有样本都使用一次
        # （假设样本数能够被批量大小整除）。这里的迭代周期个数num_epochs和学习率lr都是超参数，分别设 3 和 0.03。
        # 在实践中，大多超参数都需要通过反复试错来不断调节。当迭代周期数设的越大时，虽然模型可能更有效，但是训练时间可能过长。
        # 而有关学习率对模型的影响，我们会在后面“优化算法”一章中详细介绍。
        lr = 0.03
        num_epochs = 30
        net = linreg
        loss = squared_loss
        batch_size=10
        for epoch in range(num_epochs):  # 训练模型一共需要 num_epochs 个迭代周期。
            # 在一个迭代周期中，使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）。
            # X 和 y 分别是小批量样本的特征和标签。
            for X, y in data_iter(batch_size, features, labels):
                with autograd.record():
                    l = loss(net(X, w, b), y)  # l 是有关小批量 X 和 y 的损失。
                if len(y) < batch_size:
                    l = l * batch_size / len(y)  # 最后一个批次的数据可能小于 batch_size
                l.backward()  # 小批量的损失对模型参数求梯度。
                sgd([w, b], lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数。
            train_l = loss(net(features, w, b), labels)
            print('epoch %d, loss %f' % (epoch + 1, train_l.mean().asnumpy()))

        print(true_b,b)
        print(true_w,w)


