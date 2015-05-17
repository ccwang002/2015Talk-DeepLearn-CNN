from operator import itemgetter
import numpy as np

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
    We multiply this against a weight matrix of shape (D, M) where
    D = \prod_i d_i

    Inputs:
    x - Input data, of shape (N, d_1, ..., d_k)
    w - Weights, of shape (D, M)
    b - Biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    N = x.shape[0]
    out = x.reshape(N, -1).dot(w) + b
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
    N = x.shape[0]
    dw = x.reshape(N, -1).T.dot(dout)
    db = np.sum(dout, axis=0)
    dx = dout.dot(w.T).reshape(x.shape)
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
    out = np.where(x > 0, x, 0)
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
    x = cache
    dx = np.where(x > 0, dout, 0)
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each
    filter spans all C channels and has height HH and width HH.

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
    def convolute_per_filter(filter_th, x):
        temp_out = np.zeros((Hout, Wout), dtype=np.float64)
        for h_ix in range(Hout):
            for w_ix in range(Wout):
                h_offset = h_ix * stride
                w_offset = w_ix * stride
                receptive_field = np.s_[
                    :,                      # all channels (C)
                    h_offset:h_offset + HH,
                    w_offset:w_offset + WW
                ]
                temp_out[h_ix, w_ix] = (
                    np.sum(x[receptive_field] * w[filter_th]) + b[filter_th]
                )
        return temp_out

    pad, stride = itemgetter('pad', 'stride')(conv_param)
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    Hout = 1 + (H + 2 * pad - HH) // stride
    Wout = 1 + (W + 2 * pad - WW) // stride
    padded_x = np.pad(
        x, ((0, 0), (0, 0), (pad, pad), (pad, pad)),
        mode='constant', constant_values=0
    )
    out = np.zeros((N, F, Hout, Wout), dtype=np.float64)
    for i in range(N):
        for filter_th in range(F):
            out[i, filter_th] = convolute_per_filter(filter_th, padded_x[i])
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
    def backprop_per_filter_sample(i, f, p_x, dp_x):
        for h_ix in range(Hout):
            for w_ix in range(Wout):
                h_offset = h_ix * stride
                w_offset = w_ix * stride
                rf = np.s_[
                    :,
                    h_offset:h_offset + HH,
                    w_offset:w_offset + WW
                ]
                dp_x[rf] += w[f] * dout_per_filter[h_ix, w_ix]
                dw[f] += p_x[rf] * dout_per_filter[h_ix, w_ix]

    x, w, b, conv_param = cache
    N, F, Hout, Wout = dout.shape
    _, C, H, W = x.shape
    _, _, HH, WW = w.shape
    pad, stride = itemgetter('pad', 'stride')(conv_param)
    padded_x = np.pad(
        x, ((0, 0), (0, 0), (pad, pad), (pad, pad)),
        mode='constant', constant_values=0
    )
    dpadded_x = np.zeros(padded_x.shape)
    dw = np.zeros(w.shape)
    db = np.zeros(b.shape)
    # dout has shape (N, F, Hout, Wout)
    for i in range(N):
        for f in range(F):
            dout_per_filter = dout[i, f]
            backprop_per_filter_sample(i, f, padded_x[i], dpadded_x[i])
            db[f] += np.sum(dout_per_filter)

    # remove x padding
    dx = dpadded_x[:, :, pad:-pad, pad:-pad]
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
    def gen_pool_slice():
        for h_ix in range(HH):
            for w_ix in range(WW):
                h_offset = h_ix * stride
                w_offset = w_ix * stride
                yield (
                    h_ix, w_ix,
                    np.s_[
                        h_offset:h_offset + pool_h,
                        w_offset:w_offset + pool_w
                    ]
                )

    N, C, H, W = x.shape
    # pool_h, pool_w, stride = map(
    #     lambda k: pool_param[k],
    #     ['pool_height', 'pool_width', 'stride']
    # )
    pool_h, pool_w, stride = itemgetter(
        'pool_height', 'pool_width', 'stride'
    )(pool_param)
    HH = (H - pool_h) // stride + 1
    WW = (H - pool_w) // stride + 1
    out = np.zeros((N, C, HH, WW))
    for i in range(N):
        for c in range(C):
            x_per_ic = x[i, c]
            for h_ix, w_ix, pool_slice in gen_pool_slice():
                out[i, c, h_ix, w_ix] = np.max(x_per_ic[pool_slice])
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
    def gen_pool_slice():
        for h_ix in range(HH):
            for w_ix in range(WW):
                h_offset = h_ix * stride
                w_offset = w_ix * stride
                yield (
                    h_ix, w_ix,
                    np.s_[
                        h_offset:h_offset + pool_h,
                        w_offset:w_offset + pool_w
                    ]
                )

    x, pool_param = cache
    N, C, H, W = x.shape
    pool_h, pool_w, stride = map(
        lambda k: pool_param[k],
        ['pool_height', 'pool_width', 'stride']
    )
    HH = (H - pool_h) // stride + 1
    WW = (H - pool_w) // stride + 1
    dx = np.zeros_like(x)
    for i in range(N):
        for c in range(C):
            x_per_ic = x[i, c]
            dx_per_ic = dx[i, c]
            for h_ix, w_ix, pool_slice in gen_pool_slice():
                demux_ix = np.argmax(x_per_ic[pool_slice])
                dx_per_ic[pool_slice].flat[demux_ix] += dout[i, c, h_ix, w_ix]
    return dx


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
    - x: Input data, of shape (N, C) where x[i, j] is the score for the
      jth class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
