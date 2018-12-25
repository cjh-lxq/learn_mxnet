#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/24 15:24:48
# @Author  : 陈建宏
# @Site    : BeiJing
# @File    : 04-model_construction.py
# @Software: PyCharm
import mxnet as mx
from mxnet.gluon import nn
from mxnet import nd
# Sequential 类继承自 Block 类¶
# 我们刚刚提到，Block 类是一个通用的部件。事实上，Sequential 类继承自 Block 类。当模型的前向计算为简单串联各个层的计算时，
# 我们可以通过更加简单的方式定义模型。这正是 Sequential 类的目的：它提供add函数来逐一添加串联的 Block 子类实例，
# 而模型的前向计算就是将这些实例按添加的顺序逐一计算。
# 下面我们实现一个跟 Sequential 类有相同功能的MySequential类。这或许可以帮助你更加清晰地理解 Sequential 类的工作机制。
class MySequential(nn.Block):
    def __init__(self, **kwargs):
        super(MySequential, self).__init__(**kwargs)

    def add(self, block):
        # block 是一个 Block 子类实例，假设它有一个独一无二的名字。我们将它保存在 Block
        # 类的成员变量 _children 里，其类型是 OrderedDict。当 MySequential 实例调用
        # initialize 函数时，系统会自动对 _children 里所有成员初始化。
        self._children[block.name] = block

    def forward(self, x):
        # OrderedDict 保证会按照成员添加时的顺序遍历成员。
        for block in self._children.values():
            x = block(x)
        return x

# 虽然 Sequential 类可以使得模型构造更加简单，且不需要定义forward函数，
# 但直接继承 Block 类可以极大地拓展模型构造的灵活性。下面我们构造一个稍微复杂点的网络FancyMLP。
# 在这个网络中，我们通过get_constant函数创建训练中不被迭代的参数，即常数参数。在前向计算中，
# 除了使用创建的常数参数外，我们还使用 NDArray 的函数和 Python 的控制流，并多次调用相同的层。

class FancyMLP(nn.Block):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        # 使用 get_constant 创建的随机权重参数不会在训练中被迭代（即常数参数）。
        self.rand_weight = self.params.get_constant(
            'rand_weight', nd.random.uniform(shape=(20, 20)))
        self.dense = nn.Dense(20, activation='relu')

    def forward(self, x):
        x = self.dense(x)
        # 使用创建的常数参数，以及 NDArray 的 relu 和 dot 函数。
        x = nd.relu(nd.dot(x, self.rand_weight.data()) + 1)
        # 重用全连接层。等价于两个全连接层共享参数。
        x = self.dense(x)
        # 控制流，这里我们需要调用 asscalar 来返回标量进行比较。
        while x.norm().asscalar() > 1:
            x /= 2
        if x.norm().asscalar() < 0.8:
            x *= 10
        return x.sum()


# 由于FancyMLP和 Sequential 类都是 Block 类的子类，我们可以嵌套调用它们。
class NestMLP(nn.Block):
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Dense(64, activation='relu'),
                     nn.Dense(32, activation='relu'))
        self.dense = nn.Dense(16, activation='relu')

    def forward(self, x):
        return self.dense(self.net(x))

if __name__ == '__main__':
    with mx.Context(mx.gpu()):
        x = nd.random.uniform(shape=(2, 20))
        net1 = MySequential()
        net1.add(nn.Dense(256, activation='relu'))
        net1.add(nn.Dense(10))
        net1.initialize()
        print net1(x)

        net2 = FancyMLP()
        net2.initialize()
        print net2(x)

        # 由于FancyMLP和 Sequential 类都是 Block 类的子类，我们可以嵌套调用它们。6666
        net3 = nn.Sequential()
        net3.add(NestMLP(), nn.Dense(20), FancyMLP())
        net3.initialize()
        print net3(x)
        # 小结
        # 我们可以通过继承 Block 类来构造模型。
        # Sequential 类继承自 Block 类。
        # 虽然 Sequential 类可以使得模型构造更加简单，但直接继承 Block 类可以极大地拓展模型构造的灵活性。