
import sys
sys.path.append("..")
import utils
from mxnet import image
from mxnet.gluon import nn

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
train_data, test_data = utils.load_data_fashion_mnist(batch_size=64, resize=224)