import sys
from .. import utils
from mxnet import ndarray as nd

batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)

num_inputs = 28*28
num_outputs = 10

num_hidden = 256 # hidden layer output 256 nodes
weight_scale = 0.01

W1 = nd.random_normal(shape=(num_inputs, num_hidden), scale = weight_scale)
b1 = nd.zeros(num_hidden)

W2 = nd.random_normal(shape=(num_hidden, num_outputs), scale=weight_scale)
b2 = nd.zeros(num_outputs)

params = [W1, b2, W2, b2]
for param in params:
    param.attach_grad()

