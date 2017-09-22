from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd

net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(10)) # output is 10

net.initialize()

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.1})

# load data
def transform(data, label):
    return data.astype('float32')/255, label.astype('float32')

minist_train = gluon.data.vision.FashionMNIST(train=True, transform=transform)
minist_test = gluon.data.vision.FashionMNIST(train=False, transform=transform)

batch_size = 256
train_data = gluon.data.DataLoader(minist_train, batch_size, shuffle = True)
test_data = gluon.data.DataLoader(minist_test, batch_size, shuffle = False)

def accurancy(output, label):
    return nd.mean(output.argmax(axis=1) == label).asscalar() # output: batch_size * nums_classes
def evaluate_accurancy(data_iterator, net):
    acc = 0
    for data, label in data_iterator:
        output = net(data)
        acc += accurancy(output, label)
        #print("acc: %f" % acc)
    return acc / len(data_iterator)

for epoch in range(5):
    train_loss = 0
    train_acc = 0
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += accurancy(output, label)
    test_acc = evaluate_accurancy(test_data, net)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss / len(train_data), train_acc / len(train_data), test_acc))

