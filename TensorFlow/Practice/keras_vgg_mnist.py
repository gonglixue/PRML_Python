import os
import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np

def conv_group(input_tensor, out_dim, kernel_size, strides, stage, train_bn=True):
    x = keras.layers.Conv2D(out_dim, kernel_size, strides, padding="same",
                            name="conv2d_{}".format(stage))(input_tensor)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.BatchNormalization(name='conv_bn_{}'.format(stage))(x, training=train_bn)
    return x

class VGG_mnist():
    def __init__(self, num_classes, is_training=True):
        self.model = self.build(is_training)
        self.num_classes = num_classes


    def build(self, train_bn):
        inputs = keras.layers.Input(shape=(32, 32, 3))

        # with tf.name_scope('block_1'):
        with keras.backend.name_scope('block_1'):
            conv_1_out = conv_group(inputs, out_dim=64, kernel_size=(3, 3), strides=(1, 1),
                                    stage=1, train_bn=train_bn)

            conv_2_out = conv_group(conv_1_out, out_dim=64, kernel_size=(3, 3), strides=(1, 1),
                                    stage=2, train_bn=train_bn)

            block_1_out = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(conv_2_out)

        # with tf.name_scope('block_2'):
        with keras.backend.name_scope('block_2'):
            conv_1_out = conv_group(block_1_out, 128, (3, 3), (1, 1), 1, train_bn)
            conv_2_out = conv_group(conv_1_out, 128, (3, 3), (1, 1), 2, train_bn)
            block_2_out = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(conv_2_out)

        # with tf.name_scope('block_3'):
        with keras.backend.name_scope('block_3'):
            conv1_out = conv_group(block_2_out, 256, (3, 3), (1, 1), 1, train_bn)
            conv2_out = conv_group(conv1_out, 256, (3, 3), (1, 1), 2, train_bn)
            # conv3_out = conv_group(conv2_out, 256, (3, 3), (1, 1), 3, train_bn)
            block_3_out = keras.layers.MaxPool2D(pool_size=(2, 2))(conv2_out)

        # with tf.name_scope('block_4'):
        with keras.backend.name_scope('block_4'):
            conv1_out = conv_group(block_3_out, 512, (3, 3), (1, 1), 1, train_bn)
            conv2_out = conv_group(conv1_out, 512, (3, 3), (1, 1), 2, train_bn)
            block4_out = keras.layers.MaxPool2D(pool_size=(2, 2))(conv2_out)

        # with tf.name_scope('fully'):
        with keras.backend.name_scope('fully'):
            flatten = keras.layers.Flatten()(block4_out)
            out = keras.layers.Dense(512, activation='relu', name='dense_1')(flatten)
            out = keras.layers.BatchNormalization(name='dense_1_bn')(out)
            out = keras.layers.Dense(self.num_classes, activation='softmax')(out)

        model = keras.models.Model(inputs=inputs, outputs=out)
        return model

    def train(self, batch_size=64, learning_rate=0.01, lr_decay=1e-6, epochs=50):
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        # normalization ?

        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        datagen = keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False
        )

        datagen.fit(x_train)

        # optimizer


