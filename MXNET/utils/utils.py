from mxnet import ndarray as nd
from mxnet import gluon
import mxnet as mx

def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

def accuracy(output, label):
    return nd.mean(output.argx(aixs=1)==label).asscalar()

def evaluate_accuracy(data_iterator, net, ctx=mx.cpu()):
    acc = 0;
    for data, label in data_iterator:
        output = net(data.as_in_context(ctx))
        acc += accuracy(output, label.as_in_context(ctx))
    return acc / len(data_iterator);

def transform_mnist(data, label):
    return nd.transpose(data.astype('float32'), (2,0,1)/255, label.astype('float32'));

def load_data_fashion_mnist(batch_size, resize=None, root="~/.mxnet/datasets/fashion-mnist"):
    """download the fashion mnist dataest and then load into memory"""
    def transform_mnist(data, label):
        # transform a batch of examples
        if resize:
            n = data.shape[0]
            new_data = nd.zeros((n, resize, resize, data.shape[3]))
            for i in range(n):
                new_data[i] = image.imresize(data[i], resize, resize)
            data = new_data
        # change data from batch x height x weight x channel to batch x channel x height x weight
        return nd.transpose(data.astype('float32'), (0,3,1,2))/255, label.astype('float32')
    mnist_train = gluon.data.vision.FashionMNIST(root=root, train=True, transform=transform_mnist)
    mnist_test = gluon.data.vision.FashionMNIST(root=root, train=False, transform=transform_mnist)
    train_data = DataLoader(mnist_train, batch_size, shuffle=True)
    test_data = DataLoader(mnist_test, batch_size, shuffle=False)
    return (train_data, test_data)

def try_gpu():
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx
