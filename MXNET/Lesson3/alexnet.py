
import sys
sys.path.append("..")
import utils
#from .. import utils
from mxnet import image
from mxnet.gluon import nn
from mxnet import init
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd as autograd


def transform(data, label):
    data = image.imresize(data, 224, 224)
    return utils.transform_mnist(data, label)

net = nn.Sequential()
with net.name_scope():
    net.add(
        # first phase
        nn.Conv2D(channels=96, kernel_size=11, strides=4, activation='relu'), #卷积后输出channels 96
        nn.MaxPool2D(pool_size=3, strides=2),
        # seconde phase
        nn.Conv2D(channels=256, kernel_size=5, padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # third phase
        nn.Conv2D(channels=384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(channels=384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(channels=256, kernel_size=3, padding=1, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # forth phase
        nn.Flatten(),
        nn.Dense(4096, activation='relu'),
        nn.Dropout(0.5),
        # fifth phase
        nn.Dense(4096, activation='relu'),
        nn.Dropout(0.5),
        # sixth phase
        nn.Dense(10)
    )


batch_size = 64
train_data, test_data = utils.load_data_fashion_mnist(batch_size=64, resize=96)

ctx = utils.try_gpu()
net.initialize(ctx=ctx, init=init.Xavier());

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})
#utils.train(train_data, test_data, net, loss,
            #trainer, ctx, num_epochs=1)

for epoch in range(10):
    train_loss = 0
    train_acc = 0
    for data, label in train_data:
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data.as_in_context(ctx))
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output, label)

    test_acc = utils.evaluate_accuracy(test_data, net, ctx)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss/len(train_data),
        train_acc/len(train_data),
        test_acc
    ))



