import tensorflow as tf
import numpy as np


def add_layer(inputs, in_size, out_size, activations_fun=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    if activations_fun is None:
        outputs = Wx_plus_b
    else:
        outputs = activations_fun(Wx_plus_b)

    return outputs

def add_layer_with_namescope(inputs, in_size, out_size, activations_fun=None):
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        with tf.name_scope('wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases

    if activations_fun is None:
        outputs = Wx_plus_b
    else:
        outputs = activations_fun(Wx_plus_b)

    return outputs

def tarin_simple_mlp():

    x_data = np.linspace(-1, 1, 300, dtype=np.float32)[..., np.newaxis]
    noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
    y_data = np.square(x_data) - 0.5 + noise

    xs = tf.placeholder(tf.float32, [None, 1])
    ys = tf.placeholder(tf.float32, [None, 1])

    l1 = add_layer(xs, 1, 10, activations_fun=tf.nn.relu)
    prediction = add_layer(l1, 10, 1, activations_fun=None)

    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))

    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    for i in range(1000):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 10 == 0:
            print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))


def tarin_simple_mlp_with_namescope():
    x_data = np.linspace(-1, 1, 300, dtype=np.float32)[..., np.newaxis]
    noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
    y_data = np.square(x_data) - 0.5 + noise

    with tf.name_scope('inputs'):
        xs = tf.placeholder(tf.float32, [None, 1], name='x_in')
        ys = tf.placeholder(tf.float32, [None, 1], name='y_in')

    l1 = add_layer_with_namescope(xs, 1, 10, activations_fun=tf.nn.relu)
    prediction = add_layer_with_namescope(l1, 10, 1, activations_fun=None)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))

    with tf.name_scope('train'):
     train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    writer = tf.summary.FileWriter("logs/", sess.graph)

    sess.run(init)


    # for i in range(1000):
    #     sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    #
    #     if i % 10 == 0:
    #         print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))


if __name__ == '__main__':
    tarin_simple_mlp_with_namescope()