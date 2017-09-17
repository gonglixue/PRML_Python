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
    for i in range(0, num_examples, batch_size):
        j = nd.array(idx[i: min(i+batch_size, num_examples)])
        yield nd.take(X, j), nd.take(y,j) # ?