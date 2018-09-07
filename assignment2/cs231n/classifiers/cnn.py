from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 pool_size=2, pool_stride=2, conv_stride=1,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################

        self.params['W1'] = weight_scale * np.random.randn(num_filters,
                                                           input_dim[0],
                                                           filter_size,
                                                           filter_size)
        self.params['b1'] = np.zeros(num_filters)

        # Change this line to support non_square pooling
        pool_height = pool_width = pool_size
        # Make params available for loss() function
        self.conv_stride = conv_stride
        self.pool_param = {'pool_height': pool_height,
                           'pool_width':  pool_width,
                           'stride': pool_stride}

        # Shape of output volume of conv layer
        #out_volume_height = 1 + (input_dim[1] + 2 * (filter_size - 1) // 2 - filter_size) // conv_stride
        # assert pad ==  (filter_size - 1) // 2
        out_volume_height = 1  + (input_dim[1] - 1) // conv_stride
        out_volume_width  = 1  + (input_dim[2] - 1) // conv_stride

        # Shape of pooling output
        pool_output_height =  1 + (out_volume_height - pool_height) // pool_stride
        pool_output_width  =  1 + (out_volume_width  - pool_width ) // pool_stride
        pool_channels = num_filters
        pool_dim = pool_output_height * pool_output_width * pool_channels
        # Hidden affine layer
        self.params['W2'] = weight_scale * np.random.randn(pool_dim, hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)
        # Output affine layer
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': self.conv_stride, 'pad': (filter_size - 1) // 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        # conv - relu - 2x2 max pool - affine - relu - affine - softmax
        scores,conv_cache = conv_relu_pool_forward(X, W1, b1,
                                                   conv_param,
                                                   self.pool_param)

        scores, aff_relu_cache = affine_relu_forward(scores, W2, b2)

        scores, aff_cache = affine_forward(scores, W3, b3)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        # Compute data loss
        data_loss, dscores = softmax_loss(scores, y)

        # Compute regularization loss and add to data loss
        reg_loss = 0.0

        for key, value in self.params.items():
            #Check if key points to weight matrix and not to bias vector.
            if key.startswith('W') and key[1].isnumeric():
                W = self.params[key]
                reg_loss += np.sum(W**2)
    
        reg_loss *= 0.5 * self.reg
        loss = data_loss + reg_loss

        # Backward pass with computation of gradient
        # Output layer
        da3, dw3, db3 = affine_backward(dscores, aff_cache)
        #dL/dw3: gradient of L2 regularization loss w3^2
        dw3 += self.reg * self.params['W3']
        grads['W3'] = dw3
        grads['b3'] = db3

        # Hidden layer
        da2, dw2, db2 = affine_relu_backward(da3, aff_relu_cache)
        #dL/dw2: gradient of L2 regularization loss w2^2
        dw2 += self.reg * self.params['W2']
        grads['W2'] = dw2
        grads['b2'] = db2

        # Conv Layer
        dx, dw1, db1 = conv_relu_pool_backward(da2, conv_cache)

        #dL/dw1: gradient of L2 regularization loss w1^2
        dw1 += self.reg * self.params['W1']
        grads['W1'] = dw1
        grads['b1'] = db1
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
