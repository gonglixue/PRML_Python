from mxnet import ndarray as nd
from mxnet import autograd as ag
import random

num_inputs = 2;
num_examples = 1000

true_w = [2, -3.4];
true_b = 4.2;

X = nd.random_normal(shape = (num_examples, num_inputs)) # design matrix with 2 features
y = true_w[0] * X[:, 0] + true_w[1]*X[:, 1] + true_b
y += 0.01 * nd.random_normal(shape=y.shape)  # noise

# show training set
# print(X[0:10], y[0:10])

batch_size = 10
def data_iter():
    # generate random indices
    idx = list(range(num_examples))
    random.shuffle(idx) # randomly sort
    for i in range(0, num_examples, batch_size): #1000 examples and fetch 10 each time
        j = nd.array(idx[i: min(i+batch_size, num_examples)])
        yield nd.take(X, j), nd.take(y,j) # ?


# for data, label in data_iter():
#     print(data, label)
#     break;

w = nd.random_normal(shape= (num_inputs, 1))
b = nd.zeros((1, )) # random is also ok
params = [w, b]

for param in params:
    param.attach_grad()

# define model
def net(X):
    return nd.dot(X, w) + b # return the prediction value

# loss
def square_loss(yhat, y):
    return (yhat - y.reshape(yhat.shape)) ** 2

# optimization
def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad; # why param[:]


# training
epochs = 5  # scan 5 times for raw data
learning_rate = 0.001
for e in range(epochs):
    total_loss = 0
    for data, label in data_iter():
        with ag.record():
            output = net(data)
            loss = square_loss(output, label)  # label is the true value in traing set
        loss.backward()
        SGD(params, learning_rate)

        total_loss += nd.sum(loss).asscalar() # to float
    print("Epoch %d, average loss: %f" % (e, total_loss/num_examples))

print(true_b, b);
print(true_w, w);

