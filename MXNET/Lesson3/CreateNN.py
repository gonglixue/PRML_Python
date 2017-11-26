from mxnet import nd
from mxnet.gluon import nn
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(256, activation="relu"))
    net.add(nn.Dense(10))

print(net)
net.initialize()
# use ..

#使用nn.Block来实现上面相同的网络
class MLP(nn.Block):
    def __init__(self, **kwargs):
        #print("MLP::__init__")
        super(MLP, self).__init__(**kwargs)
        with self.name_scope():
            self.dense0 = nn.Dense(256)
            self.dense1 = nn.Dense(10)

    def forward(self, x):
        return self.dense1(nd.relu(self.dense0(x)))

net2 = MLP() # __init__
print(net2)

net2.initialize() #神经网络权重参数的初始化，这个函数继承于nn.Block
x = nd.random.uniform(shape=(4,20)) #相当于输入是4个样本，每个样本有20个feature
y = net2(x) #输出是4*10，每个样本的10个得分
print("y:")
print(y)

