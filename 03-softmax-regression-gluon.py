#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/22 12:38:58
# @Author  : 陈建宏
# @Site    : BeiJing
# @File    : 03-softmax-regression-gluon.py
# @Software: PyCharm
import gluonbook as gb
from mxnet import gluon, init,autograd,nd
import mxnet as mx
from mxnet.gluon import loss as gloss, nn
from mxnet.gluon import utils as gutils
def _get_batch(batch, ctx):
    """Return features and labels on ctx."""
    features, labels = batch
    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
    return (gutils.split_and_load(features, ctx),
            gutils.split_and_load(labels, ctx),
            features.shape[0])
def accuracy(y_hat, y):
    """Get accuracy."""
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()
def sgd(params, lr, batch_size):
    """Mini-batch stochastic gradient descent."""
    for param in params:
        param[:] = param - lr * param.grad / batch_size
def evaluate_accuracy(data_iter, net, ctx=[mx.cpu()]):
    """Evaluate accuracy of a model on the given data set."""
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc = nd.array([0])
    n = 0
    for batch in data_iter:
        features, labels, _ = _get_batch(batch, ctx)
        for X, y in zip(features, labels):
            y = y.astype('float32')
            acc += (net(X).argmax(axis=1) == y).sum().copyto(mx.cpu())
            n += y.size
        acc.wait_to_read()
    return acc.asscalar() / n
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, trainer=None):
    """Train and evaluate a model on CPU."""
    for epoch in range(1, num_epochs + 1):
        train_l_sum = 0
        train_acc_sum = 0
        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            if trainer is None:
                sgd(params, lr, batch_size)
            else:
                trainer.step(batch_size)
            train_l_sum += l.mean().asscalar()
            train_acc_sum += accuracy(y_hat, y)
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch, train_l_sum / len(train_iter),
                 train_acc_sum / len(train_iter), test_acc))
if __name__ == '__main__':
    batch_size = 256
    train_iter, test_iter = gb.load_data_fashion_mnist(batch_size)
    net = nn.Sequential()
    net.add(nn.Dense(10))
    net.initialize(init.Normal(sigma=0.01))
    loss = gloss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
    num_epochs = 5
    train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None,
                 None, trainer)