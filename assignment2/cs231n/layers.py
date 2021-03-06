from builtins import range
import numpy as np

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################

    #Reshape x into rows and perform bias trick
    x_rows = np.c_[ np.reshape(x, [x.shape[0],-1] ), np.ones(x.shape[0]) ]
    wb = np.r_[w, b[np.newaxis]]

    #Compute the forward pass, but no ReLU!
    out = x_rows.dot(wb)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    #Reshape x into rows
    x_rows = np.reshape(x, [x.shape[0],-1] )
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    df = dout # [dout/df * dL/dout = 1 * dout]
    db = np.sum(dout, axis=0) # [dout/db * dL/dout = 1 * dout ]
    
    dw = np.transpose(x_rows).dot(df) # [df/dw * dL/df = x * df]
    dx = df.dot(np.transpose(w)) # [df/dx * dL/df = w * df]
    
    # Reshape dx to the original shape of x
    # Note: dx is needed if the current layer is not the first layer
    dx = np.reshape(dx, x.shape) 
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(0.0, x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    assert dout.shape == x.shape
    dx = (x > 0.0) * dout # [df/dx * dL/df = 1(x>0) * dout]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, {}
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################

        sample_mean = np.mean(x,axis=0)

        # Compute variance manually in order to cache variables
        # for backward pass. In short, it does np.var(x, axis=0)
        numerator = x - sample_mean
        v2 = numerator**2
        sample_var = np.mean(v2,axis=0)

        # Normalize (and prepare intermediate variables for caching)
        sqt = np.sqrt( sample_var + eps )
        overx =  1. / sqt
        x_norm = overx * numerator 

        # Scale and shift
        out = gamma * x_norm + beta

        # Update running averages for test time, see doc of this function.
        running_mean = momentum * running_mean + (1. - momentum) * sample_mean
        running_var = momentum * running_var + (1. - momentum) * sample_var

        # Cache intermediate variables
        cache['xnorm'] = x_norm
        cache['gamma'] = gamma
        cache['numerator'] = numerator
        cache['1overx'] = overx
        cache['samplemean'] = sample_mean
        cache['x'] = x
        cache['sqt'] = sqt
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # Normalize
        x_norm = (x - running_mean) / np.sqrt( running_var + eps )
        # Scale and shift
        out = gamma * x_norm + beta
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################
    n = dout.shape[0]

    #Gradient of out with respect to beta
    dbeta = np.sum(dout,axis=0) # (dout * d/dbeta(x+beta)) = dout * 1
    
    # Upper branch of computational graph
    #Gradient of out w.r.t. gamma*xnorm
    dgammaxnorm = dout 

    # x_norm * gamma => Gradient swap 
    #Gradient of out w.r.t. gamma
    dgamma = np.sum(dgammaxnorm * cache['xnorm'], axis=0)

    #Gradient of out w.r.t. xnorm
    dxnorm = dgammaxnorm * cache['gamma'] 

    # Gradient of out w.r.t. numerator: dxnorm * 1overx
    # dnumerator_1.shape(N,D)
    dnumerator_1 = dxnorm * cache['1overx']

    # Lower branch of computational graph
    # Gradient of out w.r.t. 1/x: dxnorm * numerator
    d1overx = np.sum(dxnorm * cache['numerator'],axis=0)

    # Gradient of out w.r.t. sqrt(x+eps)
    dsq = -1. / (cache['sqt']**2) * d1overx 

    # Gradient of out w.r.t. sample_var (Gradient of out w.r.t. variance)
    # dsamplevar.shape(D,)
    dsamplevar = dsq / (2. * cache['sqt'])  

    # Gradient of out w.r.t. v2 (Backprop through np.mean)
    # dv2.shape(N,D)
    dv2 = dsamplevar * np.ones_like(cache['numerator']) * 1./n

    # Gradient of out w.r.t. v (Backprop through x**2 term)
    # dnumerator_2.shape(N,D)
    dnumerator_2 = dv2 * 2. * cache['numerator']

    # Merge upper and lower branch

    # Gradient of out w.r.t. samplemean
    # (Backprop through subtract-gate)
    dsamplemean = np.sum(dnumerator_1 + dnumerator_2,axis=0) * (-1.)

    # Gradient of out w.r.t. x
    # (Backprop through subtract-gate)
    dx_1 = dnumerator_1 + dnumerator_2 # * (+1)
    
    # Gradient of out w.r.t. x
    # (Backprop through np.mean)
    dx_2 = dsamplemean * np.ones_like(cache['x']) * 1./n

    #Final gradient of out w.r.t. x (Yeah!)
    dx = dx_1 + dx_2
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    N = dout.shape[0]
    
    # Gradient w.r.t. beta.
    # Just copied from other batchnorm_backward, it is good as it is.
    dbeta = np.sum(dout,axis=0)

    # Gradient w.r.t. gamma
    dgamma = np.sum( (cache['x'] - cache['samplemean'] ) / cache['sqt'] * dout , axis=0)

    # Gradient w.r.t. x
    # From https://github.com/cthorey/CS231/blob
    # /11f0521c4f7865a0005f21bb24ec29c8b00c2712/assignment2/cs231n/layers.py#L300
    #dx = (1. / N) * gamma * (var + eps)**(-1. / 2.) * (N * dout - np.sum(dout, axis=0)
    #    - (x - mu) * (var + eps)**(-1.0) * np.sum(dout * (x - mu), axis=0))
    dx = 0.1
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # (dx == dout) if the neuron was active during the forward pass.
        dx = dout * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx

def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    N = x.shape[0] 
    H = x.shape[2] 
    W = x.shape[3] 
    S = conv_param['stride'] 
    P = conv_param['pad']
    F = w.shape[0] 
    HH = w.shape[2] # Filter height
    WW = w.shape[3] # Filter width
    o_rows = 1 + (H + 2 * P - HH) // S # Output volume height
    o_cols = 1 + (W + 2 * P - WW) // S # Output volume height

    #Allocate output volume
    out = np.zeros( (N, F, o_rows, o_cols) )

    #Zero padding along width and height of input volume
    x_pad = np.pad(x,((0,),(0,),(P,),(P,)),'constant')

    for n in range(N): # Loop over images
        for f in range(F): # The fibre - Loop over all filters.
            for o_r in range(o_rows): 
                for o_c in range(o_cols):
                    out[n,f,o_r,o_c] = np.sum(
                            x_pad[n,:, S*o_r:S*o_r+HH, S*o_c:o_c*S+WW] * w[f,:,:,:]) + b[f]

    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w, b, conv_param = cache
    P = conv_param['pad']
    S = conv_param['stride']
    F, C, HH, WW = w.shape
    N, C, H, W = x.shape
    N, F, o_rows, o_cols = dout.shape

    # Allocate memory to hold the derivatives
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    # Again zero padding. We must use same input volume as we used for forw pass
    x_pad = np.pad(x,((0,),(0,),(P,),(P,)),'constant')

    # db with dimensions (F,)
    for n in range(N):
        for f in range(F):
            db[f] = np.sum(dout[:,f,:,:])

    # dw with dimensions (F,C,HH,WW)
    for f in range(F):
        for c_c in range(C):
            for f_r in range(HH):
                for f_c in range(WW):
                    # Inner gradient: Get the pixels which are connected to your weights
                    # We need a sliding window of the filter size in the input volume.
                    x_rec_field = x_pad[:, c_c, f_r:f_r+o_rows*S:S, f_c:f_c+o_cols*S:S]
                    # Multiply inner with outer gradient
                    dw[f,c_c,f_r,f_c] = np.sum( dout[:,f,:,:] * x_rec_field )

    # dx with dimensions (N, C, H, W)
    for n in range(N):
        # We need the same matrix x we used in the forward pass => Add zero padding
        x_n = np.pad( x[n], [(0, 0), (P, P), (P, P)],'constant')
        # Same goes for dx
        dx_n = np.pad( dx[n], [(0, 0), (P, P), (P, P)],'constant')

        # loop over the output volume
        for o_r in range(o_rows): 
            for o_c in range(o_cols):
                # Get the receptive field w/ *_start and *_end
                row_start = o_r * S
                row_end   = o_r * S + HH
                col_start = o_c * S
                col_end   = o_c * S + WW
                for f in range(F):
                    # In general: dx_n = w * d_out
                    # Activate  same receptive field in dx_n which was active 
                    # in forw pass. Then, get dout and multiply with every w[f].
                    dx_n[:,
                         row_start:row_end,
                         col_start:col_end] += w[f] * dout[n,f,o_r,o_c]

        dx[n] = dx_n[:,P:-P,P:-P]
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    N, C, H, W = x.shape
    HH = pool_param['pool_height']
    WW = pool_param['pool_width']
    S  = pool_param['stride']
    
    #Shape and allocation of output volume
    o_rows = 1 + (H - HH) // S
    o_cols = 1 + (W - WW) // S
    out = np.zeros((N, C, o_rows, o_cols))

    for n in range(N):
        for o_r in range(o_rows): 
            for o_c in range(o_cols):
                # Same sliding window as in conv_forward_naive()
                # But, max() is computed independently for each depth slice
                # Sth. like np.max(x[n,:,...]) doesn't work
                for c in range(C):
                    out[n,c,o_r,o_c] = np.max(x[n,
                                                c,
                                                S*o_r:S*o_r+HH,
                                                S*o_c:S*o_c+WW])
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache
    N, C, H, W = x.shape
    N, C, o_rows, o_cols = dout.shape
    HH = pool_param['pool_height']
    WW = pool_param['pool_width']
    S  = pool_param['stride']
    o_rows = 1 + (H - HH) // S
    o_cols = 1 + (W - WW) // S
    dx = np.zeros_like(x)

    for n in range(N):
        for o_r in range(o_rows): 
            for o_c in range(o_cols):
                for c in range(C):
                    #Retrieve position of max(x).This is where the gradient flows to.
                    # Get pool window
                    pool_win = x[n, c, S*o_r:S*o_r+HH, S*o_c:S*o_c+WW]
                    # Set gradient of pool window to zero except dpool[argmax(x)]
                    dpool_win = np.zeros((HH,WW))
                    idx = np.unravel_index(np.argmax(pool_win), pool_win.shape)
                    dpool_win[idx] = 1.0
                    # chain rule applied on dx[pool_window_indices]
                    dx[n, c, S*o_r:S*o_r+HH, S*o_c:S*o_c+WW] = dpool_win * dout[n,c,o_r,o_c]
    

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    # The idea is to convert x to a two-dimensional matrix in order to be feeded
    # to the vanilla BN function. The description in the Jupyter notebook gives
    # gives us a hint that the image statistics are consistent both along N
    # and along (H,W).
    #Consequently, we reshape x to (N*H*W, C):
    N, C, H, W = x.shape
    x = x.swapaxes(0,1)
    x = x.reshape(C, -1)
    x = x.T
    out, cache = batchnorm_forward(x, gamma, beta, bn_param)

    # Reshape out to the original shape of x.
    out = out.T
    out = out.reshape((C,N,H,W))
    out = out.swapaxes(0,1)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    N, C, H, W = dout.shape
    dout = dout.swapaxes(0,1)
    dout = dout.reshape(C, -1)
    dout = dout.T
    dx, dgamma, dbeta = batchnorm_backward(dout, cache)

    # Reshape out to the original shape of x.
    dx = dx.T
    dx = dx.reshape((C,N,H,W))
    dx = dx.swapaxes(0,1)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    #
    # 1. Compute the loss
    #
    # Shift all elements of x by max(x)
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    #Exponentiate all elements of x and sum the result.
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    #Final step of  $$ L_i = -f_{y_i} + log ( \sum_j e^{f_j} ) $$ 
    log_probs = shifted_logits - np.log(Z)
    #Cache probs for gradient computation
    probs = np.exp(log_probs)
    N = x.shape[0]
    #Take negative sum of the loss for each training sample.
    loss = -np.sum(log_probs[np.arange(N), y]) / N

    #
    # 2. Compute the gradient
    #
    # The partial derivative of probs[i,j] equals probs[i,j] if j!=y[i] 
    # The partial derivative of probs[i,j] equals probs[i,j] - 1 if j==y[i] 
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    # The constant factor 1/N remains when deriving the loss.
    dx /= N
    return loss, dx
