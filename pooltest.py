import numpy as np
from layers import Pooling
from utils.tools import rel_error
from utils.check_grads import check_grads_layer

import keras
from keras import layers
from keras import models
from keras import optimizers
from keras import backend as K

import warnings
warnings.filterwarnings('ignore')

inputs = np.random.uniform(size=(2, 3, 30, 30))
params = { 'pool_type': 'max',
           'pool_height': 2,
           'pool_width': 2,
           'pad': 0,
           'stride': 2,
}
layer = Pooling(params)
out = layer.forward(inputs)

keras_model = keras.Sequential()
keras_layer = layers.MaxPooling2D(pool_size=(params['pool_height'], params['pool_width']),
                                 strides=params['stride'],
                                 padding='valid',
                                 data_format='channels_first',
                                 input_shape=inputs.shape[1:])
keras_model.add(keras_layer)
sgd = optimizers.SGD(lr=0.01)
keras_model.compile(loss='mean_squared_error', optimizer='sgd')
keras_out = keras_model.predict(inputs, batch_size=inputs.shape[0])
print('Relative error (<1e-6 will be fine): ', rel_error(out, keras_out))
in_grads = np.random.uniform(size=(2, 3, 15, 15))
check_grads_layer(layer, inputs, in_grads)