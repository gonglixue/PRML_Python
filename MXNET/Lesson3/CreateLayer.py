from mxnet import nd
from mxnet.gluon import nn

class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)

    def forward(self, x):
        return x - x.mean()


# use the layer
layer = CenteredLayer()
print(layer(nd.array([1,2,3,4,5])))


# use the layer to construct network
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(128));
    net.add(nn.Dense(10));
    net.add(CenteredLayer())

net.initialize()
y = net(nd.random.uniform(shape=(4, 8)))
print(y)
print(y.mean()) # nealy zero. mxnet use float 32 bit default


# define a layer with parameter
class MyDense(nn.Block):
    def __init__(self, units, in_units, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        with self.name_scope():
            self.weight = self.params.get('weight', shape=(in_units, units))
            self.bias = self.params.get('bias', shape=(units,))

    def forward(self, x):
        linear = nd.dot(x, self.weight.data()) + self.bias.data()
        return nd.relu(linear)


my_dense = MyDense(5, in_units=10, prefix="o_my_dense_")
print(my_dense.params)

net = nn.Sequential()
with net.name_scope():
    net.add(MyDense(32, in_units=64))
    net.add(MyDense(2, in_units=32))
net.initialize()
print(net(nd.random.uniform(shape=(2, 64))))