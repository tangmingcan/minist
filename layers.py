import numpy as np 
from utils.tools import *

class Layer(object):
    """
    
    """
    def __init__(self, name):
        """Initialization"""
        self.name = name
        self.training = True  # The phrase, if for training then true
        self.trainable = False # Whether there are parameters in this layer that can be trained

    def forward(self, inputs):
        """Forward pass, reture outputs"""
        raise NotImplementedError

    def backward(self, in_grads, inputs):
        """Backward pass, return gradients to inputs"""
        raise NotImplementedError

    def update(self, optimizer):
        """Update parameters in this layer"""
        pass

    def set_mode(self, training):
        """Set the phrase/mode into training (True) or tesing (False)"""
        self.training = training

    def set_trainable(self, trainable):
        """Set the layer can be trainable (True) or not (False)"""
        self.trainable = trainable

    def get_params(self, prefix):
        """Reture parameters and gradients of this layer"""
        return None


class FCLayer(Layer):
    def __init__(self, in_features, out_features, name='fclayer', initializer=Guassian()):
        """Initialization

        # Arguments
            in_features: int, the number of inputs features
            out_features: int, the numbet of required outputs features
            initializer: Initializer class, to initialize weights
        """
        super(FCLayer, self).__init__(name=name)
        self.trainable = True

        self.weights = initializer.initialize((in_features, out_features))
        self.bias = np.zeros(out_features)

        self.w_grad = np.zeros(self.weights.shape)
        self.b_grad = np.zeros(self.bias.shape)

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, in_features)

        # Returns
            outputs: numpy array with shape (batch, out_features)
        """
        outputs = None
        #############################################################
        # code here
        #############################################################
        x = inputs.copy()
        outputs = x.reshape( x.shape[0], np.prod(x.shape[1:]) )
        outputs = outputs.dot(self.weights) + self.bias[np.newaxis, :]
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass, store gradients to self.weights into self.w_grad and store gradients to self.bias into self.b_grad

        # Arguments
            in_grads: numpy array with shape (batch, out_features), gradients to outputs
            inputs: numpy array with shape (batch, in_features), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, in_features), gradients to inputs
        """
        out_grads = None
        #############################################################
        # code here
        #############################################################
        x = inputs
        sp = x.shape
        x  = np.reshape( x, ( sp[0] , np.prod(sp[1:]) ) )
        self.w_grad = np.dot( x.T, in_grads )
        self.b_grad = np.sum( in_grads, axis = 0 )
        out_grads = np.reshape( np.dot( in_grads, self.weights.T ), sp )     
        return out_grads

    def update(self, params):
        """Update parameters (self.weights and self.bias) with new params
        
        # Arguments
            params: dictionary, one key contains 'weights' and the other contains 'bias'

        # Returns
            none
        """
        for k,v in params.items():
            if 'weights' in k:
                self.weights = v
            else:
                self.bias = v
        
    def get_params(self, prefix):
        """Return parameters (self.weights and self.bias) as well as gradients (self.w_grad and self.b_grad)
        
        # Arguments
            prefix: string, to contruct prefix of keys in the dictionary (usually is the layer-ith)

        # Returns
            params: dictionary, store parameters of this layer, one key contains 'weights' and the other contains 'bias'
            grads: dictionary, store gradients of this layer, one key contains 'weights' and the other contains 'bias'

            None: if not trainable
        """
        if self.trainable:
            params = {
                prefix+':'+self.name+'/weights': self.weights,
                prefix+':'+self.name+'/bias': self.bias
            }
            grads = {
                prefix+':'+self.name+'/weights': self.w_grad,
                prefix+':'+self.name+'/bias': self.b_grad
            }
            return params, grads
        else:
            return None

class Convolution(Layer):
    def __init__(self, conv_params, initializer=Guassian(), name='conv'):
        """Initialization

        # Arguments
            conv_params: dictionary, containing these parameters:
                'kernel_h': The height of kernel.
                'kernel_w': The width of kernel.
                'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
                'pad': The number of pixels padded to the bottom, top, left and right of each feature map. Here, pad=2 means a 2-pixel border padded with zeros.
                'in_channel': The number of input channels.
                'out_channel': The number of output channels.
            initializer: Initializer class, to initialize weights
        """
        super(Convolution, self).__init__(name=name)
        self.trainable = True
        self.kernel_h = conv_params['kernel_h'] # height of kernel
        self.kernel_w = conv_params['kernel_w'] # width of kernel
        self.pad = conv_params['pad']
        self.stride = conv_params['stride']
        self.in_channel = conv_params['in_channel']
        self.out_channel = conv_params['out_channel']

        self.weights = initializer.initialize((self.out_channel, self.in_channel, self.kernel_h, self.kernel_w))
        self.bias = np.zeros((self.out_channel))
        self.x_cols = None

        self.w_grad = np.zeros(self.weights.shape)
        self.b_grad = np.zeros(self.bias.shape)
        self.x_cols_grad = None

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            outputs: numpy array with shape (batch, out_channel, out_height, out_width)
        """
        outputs = None
        
        #############################################################
        # code here
        #############################################################
        in_batch, in_channel, in_height, in_width = inputs.shape
        filter_n, _, filter_h, filter_w = self.weights.shape
        stride, pad = self.stride, self.pad
        
        assert ( in_height + 2 * pad - filter_h ) % stride == 0, 'width doesn\'t work with current paramter setting'
        assert ( in_width + 2 * pad - filter_w ) % stride == 0, 'height doesn\'t work with current paramter setting'
        
        out_height = ( in_height + 2 * pad - filter_h ) // stride + 1
        out_width = ( in_width + 2 * pad - filter_w) // stride + 1
        
        
        outputs = np.zeros( (in_batch, filter_n, out_height, out_width), dtype=inputs.dtype )
        
        self.x_cols = im2col_indices(inputs, filter_h, filter_w, padding=pad, stride=stride)
        
        res = self.weights.reshape((self.weights.shape[0], -1)).dot(self.x_cols) + self.bias[:, np.newaxis]
        outputs = res.reshape((filter_n, out_height, out_width, in_batch))
        outputs = outputs.transpose(3, 0, 1, 2)
        
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass, store gradients to self.weights into self.w_grad and store gradients to self.bias into self.b_grad

        # Arguments
            in_grads: numpy array with shape (batch, out_channel, out_height, out_width), gradients to outputs
            inputs: numpy array with shape (batch, in_channel, in_height, in_width), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, in_channel, in_height, in_width), gradients to inputs
        """
        out_grads = None
        #############################################################
        # code here
        #############################################################
        x = inputs
        stride, pad = self.stride, self.pad
        self.x_cols = im2col_indices(inputs, self.kernel_h, self.kernel_w, padding=pad, stride=stride)
        self.b_grad = np.sum( in_grads, axis=(0, 2, 3) )
        F, _, HH, WW = self.weights.shape
        dout_reshape = np.reshape(in_grads.transpose(1,2,3,0), (F, -1))
        self.w_grad = dout_reshape.dot(self.x_cols.T).reshape(self.weights.shape)
        self.x_cols_grad = self.weights.reshape(F, -1).T.dot(dout_reshape)
        out_grads = col2im_indices(self.x_cols_grad, x.shape, field_height=HH, field_width=WW, padding=pad, stride=stride)
        return out_grads

    def update(self, params):
        """Update parameters (self.weights and self.bias) with new params
        
        # Arguments
            params: dictionary, one key contains 'weights' and the other contains 'bias'

        # Returns
            none
        """
        for k,v in params.items():
            if 'weights' in k:
                self.weights = v
            else:
                self.bias = v

    def get_params(self, prefix):
        """Return parameters (self.weights and self.bias) as well as gradients (self.w_grad and self.b_grad)
        
        # Arguments
            prefix: string, to contruct prefix of keys in the dictionary (usually is the layer-ith)

        # Returns
            params: dictionary, store parameters of this layer, one key contains 'weights' and the other contains 'bias'
            grads: dictionary, store gradients of this layer, one key contains 'weights' and the other contains 'bias'

            None: if not trainable
        """
        if self.trainable:
            params = {
                prefix+':'+self.name+'/weights': self.weights,
                prefix+':'+self.name+'/bias': self.bias
            }
            grads = {
                prefix+':'+self.name+'/weights': self.w_grad,
                prefix+':'+self.name+'/bias': self.b_grad
            }
            return params, grads
        else:
            return None

class ReLU(Layer):
    def __init__(self, name='relu'):
        """Initialization
        """
        super(ReLU, self).__init__(name=name)

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array

        # Returns
            outputs: numpy array
        """
        outputs = np.maximum(0, inputs)
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass

        # Arguments
            in_grads: numpy array, gradients to outputs
            inputs: numpy array, same with forward inputs

        # Returns
            out_grads: numpy array, gradients to inputs 
        """
        inputs_grads = (inputs >=0 ) * in_grads
        out_grads = inputs_grads
        return out_grads


# TODO: add padding
class Pooling(Layer):
    def __init__(self, pool_params, name='pooling'):
        """Initialization

        # Arguments
            pool_params is a dictionary, containing these parameters:
                'pool_type': The type of pooling, 'max' or 'avg'
                'pool_h': The height of pooling kernel.
                'pool_w': The width of pooling kernel.
                'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
                'pad': The number of pixels padded to the bottom, top, left and right of each feature map. Here, pad=2 means a 2-pixel border padded with zeros.
        """
        super(Pooling, self).__init__(name=name)
        self.pool_type = pool_params['pool_type']
        self.pool_height = pool_params['pool_height']
        self.pool_width = pool_params['pool_width']
        self.stride = pool_params['stride']
        self.pad = pool_params['pad']

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            outputs: numpy array with shape (batch, in_channel, out_height, out_width)
        """
        outputs = None
        #############################################################
        # code here
        #############################################################
        N,C,H,W = inputs.shape
        pool_type,pool_height,pool_width,stride,pad = self.pool_type, self.pool_height,self.pool_width,self.stride, self.pad
        assert ( H + 2 * pad - pool_height ) % stride == 0, 'width doesn\'t work with current paramter setting'
        assert ( W + 2 * pad - pool_width ) % stride == 0, 'height doesn\'t work with current paramter setting'
        
        out_height = ( H + 2 * pad - pool_height ) // stride + 1
        out_width = ( W + 2 * pad - pool_width ) // stride + 1

        outputs = np.zeros((N,C,out_height,out_width))
        for ii,i in enumerate(range(0,H+2*pad-pool_height+stride,stride)):
            for jj,j in enumerate(range(0,W+2*pad-pool_width+stride,stride)):
                if pool_type=='max':
                    outputs[:,:,ii,jj] = np.amax( inputs[:, :, i:i+pool_height,j:j+pool_width].reshape(N, C, -1), axis=2 )
                elif pool_type=='avg':
                    outputs[:,:,ii,jj] = np.average( inputs[:, :, i:i+pool_height,j:j+pool_width].reshape(N, C, -1), axis=2 )
                else:
                    raise Exception("Invalid pool type parameter!")
                
        return outputs
        
    def backward(self, in_grads, inputs):
        """Backward pass

        # Arguments
            in_grads: numpy array with shape (batch, in_channel, out_height, out_width), gradients to outputs
            inputs: numpy array with shape (batch, in_channel, in_height, in_width), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, in_channel, in_height, in_width), gradients to inputs
        """
        out_grads = None
        #############################################################
        # code here
        #############################################################
        N,C,H,W = inputs.shape
        pool_type,pool_height,pool_width,stride,pad = self.pool_type, self.pool_height,self.pool_width,self.stride, self.pad
        assert ( H + 2 * pad - pool_height ) % stride == 0, 'width doesn\'t work with current paramter setting'
        assert ( W + 2 * pad - pool_width ) % stride == 0, 'height doesn\'t work with current paramter setting'
        out_grads = np.zeros_like(inputs)

        
        if pool_type=='max':
            for ii, i in enumerate(range(0,H+2*pad-pool_height+stride, stride)):
                for jj,j in enumerate(range(0,W+2*pad-pool_width+stride,stride)):
                    max_idx= np.argmax( inputs[:, :, i:i+pool_height,j:j+pool_width].reshape(N, C, -1), axis=2)
                    max_cols = np.remainder(max_idx, pool_width) + j
                    max_rows = max_idx // pool_width + i
                    
                    for n in range(N):
                        for c in range(C):
                            out_grads[n,c,max_rows[n,c],max_cols[n,c]]+= in_grads[n,c,ii,jj]
        elif pool_type=='avg':
            for ii, _ in enumerate(range(0,H+2*pad-pool_height+stride, stride)):
                for jj,_ in enumerate(range(0,W+2*pad-pool_width+stride,stride)):
                    for k in range(ii*stride,min(ii*stride+pool_height,H)):
                        for l in range(jj*stride,min(jj*stride+pool_width,W)):
                            for n in range(N):
                                for c in range(C):
                                    out_grads[n,c,k,l]+=in_grads[n,c,ii,jj]/(pool_height*pool_width)
        out_grads=out_grads.reshape(N,C,H,W)
        return out_grads

class Dropout(Layer):
    def __init__(self, ratio, name='dropout', seed=None):
        """Initialization

        # Arguments
            ratio: float [0, 1], the probability of setting a neuron to zero
            seed: int, random seed to sample from inputs, so as to get mask. (default as None)
        """
        super(Dropout, self).__init__(name=name)
        self.ratio = ratio
        self.mask = None
        self.seed = seed

    def forward(self, inputs):
        """Forward pass (Hint: use self.training to decide the phrase/mode of the model)

        # Arguments
            inputs: numpy array

        # Returns
            outputs: numpy array
        """
        outputs = None
        #############################################################
        # code here
        #############################################################
        p,mode=self.ratio, self.training
        np.random.seed(self.seed)
        
        if mode is True:
            mask=(np.random.rand(*inputs.shape)<p)/p
            outputs= inputs * mask
        else:
            mask=None
            outputs= inputs
        self.mask=mask
        outputs=outputs.astype(inputs.dtype,copy=False)
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass

        # Arguments
            in_grads: numpy array, gradients to outputs
            inputs: numpy array, same with forward inputs

        # Returns
            out_grads: numpy array, gradients to inputs 
        """
        out_grads = None
        #############################################################
        # code here
        #############################################################
        mask,mode=self.mask,self.training
        if mode is True:
            out_grads=in_grads*mask
        else:
            out_grads=in_grads
        
        return out_grads

class Flatten(Layer):
    def __init__(self, name='flatten', seed=None):
        """Initialization
        """
        super(Flatten, self).__init__(name=name)

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            outputs: numpy array with shape (batch, in_channel*in_height*in_width)
        """
        batch = inputs.shape[0]
        outputs = inputs.copy().reshape(batch, -1)
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass

        # Arguments
            in_grads: numpy array with shape (batch, in_channel*in_height*in_width), gradients to outputs
            inputs: numpy array with shape (batch, in_channel, in_height, in_width), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, in_channel, in_height, in_width), gradients to inputs 
        """
        out_grads = in_grads.copy().reshape(inputs.shape)
        return out_grads

    
def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,stride)
    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols

def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,stride=1):
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0

    out_height = (H + 2 * padding - field_height) // stride + 1
    out_width = (W + 2 * padding - field_width) // stride + 1

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)
