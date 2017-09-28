from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd
import matplotlib.pyplot as plt
import sys
import random

def transform(data, label):
    return data.astype('float32')/255, label.astype('float32')

minist_train = gluon.data.vision.FashionMNIST(train=True, transform=transform)
minist_test = gluon.data.vision.FashionMNIST(train=False, transform=transform)

def show_image(images):
    n = images.shape[0] # nums of images
    _, figs = plt.subplots(1, n, figsize=(15, 15))
    for i in range(n):
        figs[i].imshow(images[i].reshape((28, 28)).asnumpy())
        figs[i].axes.get_xaxis().set_visible(False)
        figs[i].axes.get_yaxis().set_visible(False)
    plt.show()

def get_text_labels(label):
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress,', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in label]

# data, label = minist_train[0:9]
# show_image(data)
# print(get_text_labels(label))

# load data
batch_size = 256
train_data = gluon.data.DataLoader(minist_train, batch_size, shuffle = True)
test_data = gluon.data.DataLoader(minist_test, batch_size, shuffle = False)

# initialize parameters
num_inputs = 784 # 28*28 num of features
num_outputs = 10 # num of class

W = nd.random_normal(shape=(num_inputs, num_outputs))
b = nd.random_normal(shape=num_outputs) # row vec

params = [W, b]

for param in params:
    param.attach_grad()

def softmax(X):
    exp = nd.exp(X)
    partition = exp.sum(axis = 1, keepdims=True) # return (nrows, 1) matrix
    return exp / partition

def net(X):
    return softmax(nd.dot(X.reshape((-1, num_inputs)), W) + b)
    # the second dim is num_inputs, the first dim will be automatically calculated

def cross_entropy(yhat, y):
    return - nd.pick(nd.log(yhat), y)

def accurancy(output, label):
    return nd.mean(output.argmax(axis=1) == label).asscalar() # output: batch_size * nums_classes
# output.argmax(axis=1) 返回一个 1*10 矩阵，每个元素代表output每行最大值的列号
def evaluate_accurancy(data_iterator, net):
    acc = 0
    for data, label in data_iterator:
        output = net(data)
        acc += accurancy(output, label)
        #print("acc: %f" % acc)
    return acc / len(data_iterator)

def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad;


learning_rate = 0.05
for epoch in range(10):
    train_loss = 0
    train_acc = 0
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = cross_entropy(output, label)
        loss.backward()
        SGD(params, learning_rate/batch_size) #将梯度做平均，这样学习率对batch_size不会那么敏感

        train_loss += nd.mean(loss).asscalar()
        train_acc += accurancy(output, label)

    test_acc = evaluate_accurancy(test_data, net)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (epoch, train_loss/len(train_data), train_acc/len(train_data),
                                                             test_acc)
          )

data, label = minist_test[0:9]
show_image(data)
print('true labels')
print(get_text_labels(label))

predicted_labels = net(data).argmax(axis=1)
print('predicted labels')
print(get_text_labels(predicted_labels.asnumpy()))