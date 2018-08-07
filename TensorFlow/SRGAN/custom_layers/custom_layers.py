
import numpy as np
from tensorflow import keras
from keras.engine.topology import Layer

class vgg_input_norm(Layer):
    def __init__(self, type="vgg", value=120):
        self.type = type
        self.value = value

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        if self.type == "gan":
            return (x - self.value) / self.value    # [0, 255] -> [-1, 1]
        else:
            x = x - self.value

        return x

    def get_output_shape_for(self, input_shape):
        return input_shape


