import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

import argparse
import sys
import tempfile

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variab(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def deepnn(x):
    '''
    build the graph for a deep net
    :param x: [N_examples, 784]
    :return: a tuple(y, keep_prob). y:[n_examples, 10). keep_prob is a scalar placeholder for the probability of dropout
    '''
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    # conv 1
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variab([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # pooling layer
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # conv2
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variab([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # second pooling layer
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # fully connected layer 1
    # 7 * 7 * 64
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7*7*64, 1024])
        b_fc1 = bias_variab([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # drop out
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # output
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variab([10])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_conv, keep_prob


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def custom_softmax_cross_entropy_logits():
    # our NN's output
    logits = tf.constant([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    # step1:do softmax
    y = tf.nn.softmax(logits)
    # true label
    y_ = tf.constant([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    # step2:do cross_entropy
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    # do cross_entropy just one step
    cross_entropy2 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits, y_))  # dont forget tf.reduce_sum()!!

    with tf.Session() as sess:
        softmax = sess.run(y)
        c_e = sess.run(cross_entropy)
        c_e2 = sess.run(cross_entropy2)
        print("step1:softmax result=")
        print(softmax)
        print("step2:cross_entropy result=")
        print(c_e)
        print("Function(softmax_cross_entropy_with_logits) result=")
        print(c_e2)

def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10]) # groundtruth

    y_conv, keep_prob = deepnn(x)   # build graph

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
        # logits是做softmax前的输出值。softmax_cross_entropy_with_logits会对logits做softmax
        # 所以网络里不需要softmax
        # softmax_cross_entropy: 返回的是一个向量, 每个元素是y_i^ * log(y_i). reduce_sum之后才是交叉熵

    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correction_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))    # [num_examples, 1]
        correction_prediction = tf.cast(correction_prediction, tf.float32)  # type transfer

    accuracy = tf.reduce_mean(correction_prediction)

    graph_location = "temp"
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(20000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x:batch[0], y_:batch[1], keep_prob: 1.0
                })
                print('step %d: training acc %g' % (i, train_accuracy))

        print('test acc %g' % (accuracy.eval(feed_dict={
            x:mnist.test.images, y_:mnist.test.labels, keep_prob: 1.0
        })))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./dataset')

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

