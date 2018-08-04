import os
import sys
# import tensorflow as tf
from tensorflow import keras
import numpy as np

def conv_group(input_tensor, out_dim, kernel_size, strides, scope_name, stage, train_bn=True):
    x = keras.layers.Conv2D(out_dim, kernel_size, strides, padding="same",
                            name="{}_conv2d_{}".format(scope_name, stage))(input_tensor)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.BatchNormalization(name='{}_conv_bn_{}'.format(scope_name, stage))(x, training=train_bn)
    return x

class VGG_cifar():
    def __init__(self, num_classes, is_training=True):
        self.num_classes = num_classes
        self.model = self.build(is_training)

    def build(self, train_bn):
        inputs = keras.layers.Input(shape=(32, 32, 3))

        # with tf.name_scope('block_1'):
        with keras.backend.name_scope('block_1'):
            conv_1_out = conv_group(inputs, out_dim=64, kernel_size=(3, 3), strides=(1, 1),
                                    scope_name='block_1', stage=1, train_bn=train_bn)

            conv_2_out = conv_group(conv_1_out, out_dim=64, kernel_size=(3, 3), strides=(1, 1),
                                    scope_name='block_1', stage=2, train_bn=train_bn)

            block_1_out = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(conv_2_out)

        # with tf.name_scope('block_2'):
        with keras.backend.name_scope('block_2'):
            conv_1_out = conv_group(block_1_out, 128, (3, 3), (1, 1), 'block_2', 1, train_bn)
            conv_2_out = conv_group(conv_1_out, 128, (3, 3), (1, 1), 'block_2', 2, train_bn)
            block_2_out = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(conv_2_out)

        # with tf.name_scope('block_3'):
        with keras.backend.name_scope('block_3'):
            conv1_out = conv_group(block_2_out, 256, (3, 3), (1, 1), 'block_3', 1, train_bn)
            conv2_out = conv_group(conv1_out, 256, (3, 3), (1, 1), 'block_3', 2, train_bn)
            # conv3_out = conv_group(conv2_out, 256, (3, 3), (1, 1), 3, train_bn)
            block_3_out = keras.layers.MaxPool2D(pool_size=(2, 2))(conv2_out)

        # with tf.name_scope('block_4'):
        with keras.backend.name_scope('block_4'):
            conv1_out = conv_group(block_3_out, 512, (3, 3), (1, 1), 'block_4', 1, train_bn)
            conv2_out = conv_group(conv1_out, 512, (3, 3), (1, 1), 'block_4', 2, train_bn)
            block4_out = keras.layers.MaxPool2D(pool_size=(2, 2))(conv2_out)

        # with tf.name_scope('fully'):
        with keras.backend.name_scope('fully'):
            flatten = keras.layers.Flatten(name='flatten')(block4_out)
            out = keras.layers.Dense(512, activation='relu', name='dense_1')(flatten)
            out = keras.layers.BatchNormalization(name='dense_1_bn')(out)
            out = keras.layers.Dense(self.num_classes, activation='softmax', name='output')(out)

        model = keras.models.Model(inputs=inputs, outputs=out)
        return model

    def normalize(self, X_train, X_test=None):
        # this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(X_train, axis=(0, 1, 2, 3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train - mean) / (std + 1e-7)

        if X_test is None:
            return X_train
        else:
            X_test = (X_test - mean) / (std + 1e-7)
            return X_train, X_test

    def train(self, batch_size=64, learning_rate=0.005, lr_decay=1e-6, epochs=20, ckpt_dir="./cifar10_checkpoints"):
        if not os.path.exists(ckpt_dir):
            os.mkdir(ckpt_dir)

        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        x_train = x_train.astype('float32')
        # normalization ?
        x_train = self.normalize(x_train)
        y_train = keras.utils.to_categorical(y_train, self.num_classes)


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
        sgd = keras.optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=1)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        lr_drop = 20
        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))
        reduce_lr_callback = keras.callbacks.LearningRateScheduler(lr_scheduler)
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir='cifar10_log', histogram_freq=0, write_grads=True, write_images=True)
        ckpt_file_path = os.path.join(ckpt_dir, "cifar-ckpt-improve-{epoch:02d}-{loss:.2f}.hdf5")
        ckpt_callback = keras.callbacks.ModelCheckpoint(ckpt_file_path, monitor='loss',
                                                        save_weights_only=True, save_best_only=True, verbose=1)

        self.model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                 steps_per_epoch=x_train.shape[0] // batch_size,
                                 epochs=epochs,
                                 callbacks=[reduce_lr_callback, tensorboard_callback, ckpt_callback],
                                 verbose=1)

        self.model.save_weights('cifar-final.h5')


if __name__ == '__main__':
    classifier = VGG_cifar(10, True)
    classifier.train()
