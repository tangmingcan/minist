import numpy as np
from layers import Convolution
from utils.tools import rel_error

import keras
from keras import layers
from keras import models
from keras import optimizers
from keras import backend as K

import warnings
warnings.filterwarnings('ignore')

inputs = np.random.uniform(size=(10, 3, 30, 30))
params = { 'kernel_h': 4,
          'kernel_w': 4,
          'pad': 0,
          'stride': 2,
          'in_channel': inputs.shape[1],
          'out_channel': 64,
}
layer = Convolution(params)
out = layer.forward(inputs)

keras_model = keras.Sequential()
keras_layer = layers.Conv2D(filters=params['out_channel'],
                            kernel_size=(params['kernel_h'], params['kernel_w']),
                            strides=(params['stride'], params['stride']),
                            padding='valid',
                            data_format='channels_first',
                            input_shape=inputs.shape[1:])
keras_model.add(keras_layer)
sgd = optimizers.SGD(lr=0.01)
keras_model.compile(loss='mean_squared_error', optimizer='sgd')
weights = np.transpose(layer.weights, (2, 3, 1, 0))
keras_layer.set_weights([weights, layer.bias])
keras_out = keras_model.predict(inputs, batch_size=inputs.shape[0])
print('Relative error (<1e-6 will be fine): ', rel_error(out, keras_out))