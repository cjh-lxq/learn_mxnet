#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/20 9:39:45
# @Author  : 陈建宏
# @Site    : BeiJing
# @File    : 00_hello_world.py.py
# @Software: PyCharm Community Edition
from __future__ import division
from __future__ import print_function
from mxnet import autograd, nd
from mxnet.gluon import data as gdata
from mxnet.gluon import nn
from mxnet import init
from mxnet.gluon import loss as gloss
from mxnet import gluon
import mxnet as mx
def main():
    debug_flage = True
    num_inputs = 2
    num_examples = 1000
    true_w = [2, -3.4]
    true_b = 4.2
    features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    labels += nd.random.normal(scale=0.01, shape=labels.shape)
    # Gluon 提供了data模块来读取数据。由于data常用作变量名，我们将导入的data模块用添加了 Gluon 首字母的假名gdata代替。
    # 在每一次迭代中，我们将随机读取包含 10 个数据样本的小批量。
    batch_size = 10
    # 将训练数据的特征和标签组合。
    dataset = gdata.ArrayDataset(features, labels)
    # 随机读取小批量。
    data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
    if debug_flage:
        for X, y in data_iter:
            print(X, y)
            break
    # 在上一节从零开始的实现中，我们需要定义模型参数，并使用它们一步步描述模型是怎样计算的。
    # 当模型结构变得更复杂时，这些步骤将变得更加繁琐。其实，Gluon 提供了大量预定义的层，这使我们只需关注使用哪些层来构造模型。
    # 下面将介绍如何使用 Gluon 更简洁地定义线性回归。

    # 首先，导入nn模块。实际上，“nn”是 neural networks（神经网络）的缩写。顾名思义，该模块定义了大量神经网络的层。
    # 我们先定义一个模型变量net，它是一个 Sequential 实例。在 Gluon 中，Sequential 实例可以看作是一个串联各个层的容器。
    # 在构造模型时，我们在该容器中依次添加层。当给定输入数据时，容器中的每一层将依次计算并将输出作为下一层的输入。
    net = nn.Sequential()
    # 回顾图 3.1 中线性回归在神经网络图中的表示。作为一个单层神经网络，线性回归输出层中的神经元和输入层中各个输入完全连接。
    # 因此，线性回归的输出层又叫全连接层。在 Gluon 中，全连接层是一个Dense实例。我们定义该层输出个数为 1。
    # 值得一提的是，在 Gluon 中我们无需指定每一层输入的形状，例如线性回归的输入个数。当模型看见数据时，例如后面执行net(X)时，
    # 模型将自动推断出每一层的输入个数。我们将在之后“深度学习计算”一章详细介绍这个机制。Gluon 的这一设计为模型开发带来便利。
    net.add(nn.Dense(1000, activation='relu'))
    net.add(nn.Dense(1000, activation='relu'))
    net.add(nn.Dense(1000, activation='relu'))
    net.add(nn.Dense(1000, activation='relu'))
    net.add(nn.Dense(1000, activation='relu'))
    net.add(nn.Dense(1000, activation='relu'))
    net.add(nn.Dense(1))
    # 在使用net前，我们需要初始化模型参数，例如线性回归模型中的权重和偏差。我们从 MXNet 导入initializer模块。
    # 该模块提供了模型参数初始化的各种方法。这里的init是initializer的缩写形式。我们通过init.Normal(sigma=0.01)
    # 指定权重参数每个元素将在初始化时随机采样于均值为 0 标准差为 0.01 的正态分布。偏差参数默认会初始化为零。
    net.initialize(init.Normal(sigma=0.01))
    # 在 Gluon 中，loss模块定义了各种损失函数。我们用假名gloss代替导入的loss模块，
    # 并直接使用它所提供的平方损失作为模型的损失函数。
    loss = gloss.L2Loss()  # 平方损失又称 L2 范数损失。
    # 同样，我们也无需实现小批量随机梯度下降。在导入 Gluon 后，我们创建一个Trainer实例，
    # 并指定学习率为 0.03 的小批量随机梯度下降（sgd）为优化算法。
    # 该优化算法将用来迭代net实例所有通过add函数嵌套的层所包含的全部参数。这些参数可以通过collect_params函数获取。
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.001})
    # 在使用 Gluon 训练模型时，我们通过调用Trainer实例的step函数来迭代模型参数。
    # 上一节中我们提到，由于变量l是长度为batch_size的一维 NDArray，执行l.backward()等价于执行l.sum().backward()。
    # 按照小批量随机梯度下降的定义，我们在step函数中指明批量大小，从而对批量中样本梯度求平均。
    num_epochs = 30
    for epoch in range(1, num_epochs + 1):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        l = loss(net(features), labels)
        print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))
    #  下面我们分别比较学到的和真实的模型参数。我们从net获得需要的层，并访问其权重（weight）和偏差（bias）。学到的和真实的参数很接近。
    # dense = net2[0]
    # print(true_w, dense.weight.data())
    # print(true_b, dense.weight.data())
if __name__ == '__main__':
    with mx.Context(mx.gpu()):
        main()
