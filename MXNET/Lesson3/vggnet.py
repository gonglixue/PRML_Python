from mxnet.gluon import nn
from mxnet import nd

import sys
sys.path.append("..")
import utils
from mxnet import gluon
from mxnet import init

# 传入的channels参数是指卷积后的通道数
def vgg_block(num_convs, channels):
    out = nn.Sequential()
    with out.name_scope():
        for _ in range(num_convs):
            out.add(nn.Conv2D(channels=channels, kernel_size=3, padding=1, activation='relu'))

    out.add(nn.MaxPool2D(pool_size=2, strides=2))
    return out

def vgg_stack(architecture):
    out = nn.Sequential()
    with out.name_scope():
        for (num_convs, channels) in architecture:
            out.add(vgg_block(num_convs, channels))

    return out

# 一个最简单的VGG11
def vgg11():
    num_outputs = 10
    architecture = ((1,64), (1,128), (2,256), (2,512), (2,512))
    vgg11_net = nn.Sequential()

    with vgg11_net.name_scope():
        vgg11_net.add(vgg_stack(architecture))
        vgg11_net.add(nn.Flatten()) #
        vgg11_net.add(nn.Dense(4096, activation='relu'))
        vgg11_net.add(nn.Dropout(0.5))
        vgg11_net.add(nn.Dense(4096, activation='relu'))
        vgg11_net.add(nn.Dropout(0.5))
        vgg11_net.add(nn.Dense(num_outputs))

    return vgg11_net

def vgg19():
    num_outputs = 10
    architecture = ((2,64), (2,128), (4,256), (4,512), (4,512))
    net = nn.Sequential()

    with net.name_scope():
        net.add(vgg_stack(architecture))
        net.add(nn.Flatten())
        net.add(nn.Dense(4096, activation='relu'))
        net.add(nn.Dropout(0.5))
        net.add(nn.Dense(4096, activation='relu'))
        net.add(nn.Dropout(0.5))
        net.add(nn.Dense(num_outputs))

    return net


def train_vgg11(train_data, test_data):
    print("begin train vgg11")
    ctx = utils.try_all_gpus()
    net = vgg11()
    net.initialize(ctx=ctx, init=init.Xavier())

    soft_max_cross = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.05})
    utils.train(train_data, test_data, net, soft_max_cross, trainer, ctx, num_epochs=10);


def train_vgg19(train_data, test_data):
    print("begin train vgg19")
    ctx = utils.try_all_gpus()
    net = vgg19()
    net.initialize(ctx=ctx, init=init.Xavier())

    soft_max_cross = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.05})
    utils.train(train_data, test_data, net, soft_max_cross, trainer, ctx, num_epochs=10);

train_data, test_data = utils.load_data_fashion_mnist(
    batch_size=64, resize=96)
train_vgg11(train_data, test_data)