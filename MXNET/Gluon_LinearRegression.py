from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon
import random

num_inputs = 2;
num_examples = 1000

true_w = [2, -3.4];
true_b = 4.2;

X = nd.random_normal(shape = (num_examples, num_inputs)) # design matrix with 2 features
y = true_w[0] * X[:, 0] + true_w[1]*X[:, 1] + true_b
y += 0.01 * nd.random_normal(shape=y.shape)  # noise

batch_size = 10
dataset = gluon.data.ArrayDataset(X, y)
data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True)

# define model
net = gluon.nn.Sequential()
net.add(gluon.nn.Dense(1))

# initial parameters
net.initialize() # default randomly initialize

# loss
square_loss = gluon.loss.L2Loss()

# optimization
trainer = gluon.Trainer(
    net.collect_params(), 'sgd', {'learning_rate': 0.1}
)

# training
epochs = 5
for e in range(epochs):
    total_loss = 0
    for data, label in data_iter:
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        trainer.step(batch_size) # ?? use batch_size for what?
        total_loss += nd.sum(loss).asscalar()
    print("Epoch %d: average loss: %f" % (e, total_loss/num_examples))

dense = net[0]
print(true_w, dense.weight.data())
print(true_b, dense.bias.data())