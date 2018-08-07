import sys

from tensorflow import keras
from keras.applications import VGG19
import numpy as np

from custom_layers import custom_layers



class VGGNetwork():
    def __init__(self, img_length=384, vgg_weight=1.0):
        self.img_length = 384
        self.vgg_weight = vgg_weight
        self.vgg_layers = None

    def append_vgg_network(self, x_in, true_X_input, pre_train=False):
        x = keras.layers.concatenate([x_in, true_X_input], axis=0)

        # normalization the inputs via custom VGG normalization layer
        x = custom_layers.vgg_input_norm(name="normalize_vgg")(x)

        # build VGG layer
        x = keras.layers.Conv2D()


class SRGAN():
    def __init__(self):
        self.channels = 3
        self.lr_height = self.lr_width = 64
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)  # tensorflow channel last

        self.hr_height = self.lr_height * 4
        self.hr_width = self.hr_width * 4
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)

        self.n_resudual_blocks = 16


    def build_vgg19(self):
        '''
        see architecture at : https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py
        :return:
        '''
        # set keras-model outputs to the output of last conv layer
        vgg = VGG19(weights="imagenet")
        vgg.outputs = [vgg.layers[9].output]

        img = keras.layers.Input(shape=self.hr_shape)
        img_features = vgg(img)

        return keras.models.Model(img, img_features)

    def build_generator(self):
        def residual_block(layer_input, id):
            x = keras.layers.Conv2D(64, 3, 1, padding='same', name="gen_res{}_conv1".format(id))(layer_input)
            x = keras.layers.Activation('relu')(x)
            x = keras.layers.BatchNormalization(momentum=0.8, name="gen_res{}_bn1".format(id))(x)

            x = keras.layers.Conv2D(64, 3, 1, padding='same', name="gen_res{}_conv2".format(id))(x)
            x = keras.layers.BatchNormalization(momentum=0.8, name="gen_res{}_bn2".format(id))(x)
            x = keras.layers.Add()([x, layer_input])

            return x

        input_low_res = keras.layers.Input(shape=self.lr_shape)

        # stage before residual blocks
        stage1 = keras.layers.Conv2D(64, 9, 1, padding='same', name="gen_pre_conv")(input_low_res)
        stage1 = keras.layers.Activation('relu')(stage1)

        # multiple residual blocks
        res_output = residual_block(stage1, id=0)
        for i in range(self.n_resudual_blocks-1):
            res_output = residual_block(res_output, id=i+1)

        # stage after residual blocks
        stage2 = keras.layers.Conv2D(64, 3, 1, padding='same', name="gen_post_conv")(res_output)
        stage2 = keras.layers.BatchNormalization(momentum=0.8, name="gen_post_bn")(stage2)



