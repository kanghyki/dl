import numpy as np
from numpy.typing import NDArray

# functions

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    orig_shape = x.shape
    if x.ndim == 1:
        x = x.reshape(1, -1)

    # Numerical stability
    x = x - x.max(axis=1, keepdims=True)

    exp_x = np.exp(x)
    softmax_x = exp_x / exp_x.sum(axis=1, keepdims=True)

    if orig_shape == softmax_x.shape:
        return softmax_x
    else:
        return softmax_x.reshape(orig_shape)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

# layers

class Sigmoid:
    def __init__(self):
        self.params = []
        self.grads = []

        self.out: NDArray = None

    def forward(self, x):
        self.out = sigmoid(x)
        return self.out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

class Relu:
    def __init__(self):
        self.params = []
        self.grads = []
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        self.x = x
        return np.matmul(x, W) + b

    def backward(self, dout):
        W, _ = self.params
        dx = np.matmul(dout, W.T)
        dW = np.matmul(self.x.T, dout)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx

class SoftMaxWithCrossEntropyError:
    def __init__(self):
        self.params = []
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.y = softmax(x)
        self.t = t
        # If the correct answer label is a one-hot vector,
        # convert it to an index of the correct answer
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy_error(y=self.y, t=self.t)

        return loss

    def backward(self, dout):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size
        return dx

class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None
        pass

    def forward(self, x):
        W = self.params
        self.x = x
        return np.matmul(x, W)

    def backward(self, dout):
        W = self.params

        # Y(dout) = N*H
        # W = D*H
        # W.T = H*D
        # dx = N*D
        dx = np.matmul(dout, W.T)

        # x = N*D
        # x.T = D*N
        # dw = D*H
        dW = np.matmul(self.x.T, dout)

        self.grads[0][...] = dW

        return dx

class BatchNorm:
    def __init__(self, D):
        self.gamma = np.ones(D)
        self.beta = np.zeros(D)
        self.params = [self.gamma, self.beta]
        self.grads = [np.zeros(D), np.zeros(D)]
        self.eps = 1e-7
        self.cache:tuple = None
        self.train_mode = True
        self.running_mean = np.zeros(D)
        self.running_var = np.ones(D)
        self.momentum = 0.9

    def forward(self, x):
        if self.train_mode:
            N, _ = x.shape
            mu = np.sum(x, axis=0) / N
            xmu = x - mu
            sq = xmu**2
            var = np.sum(sq, axis=0) / N
            sqrtvar = np.sqrt(var + self.eps)
            ivar = 1. / sqrtvar
            xhat = xmu * ivar
            gammax = self.gamma * xhat
            out = gammax + self.beta
            self.cache = (xhat, xmu, ivar, sqrtvar, var)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

            return out
        else:
            xhat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * xhat + self.beta
            return out

    def backward(self, dout):
        N, D = dout.shape
        gamma = self.gamma
        xhat, xmu, ivar, sqrtvar, var = self.cache

        dbeta = np.sum(dout, axis=0)
        dgammax = dout

        dgamma = np.sum(dgammax*xhat, axis=0)
        self.grads[0][...] = dgamma
        self.grads[1][...] = dbeta

        dxhat = dgammax * gamma
        divar = np.sum(dxhat * xmu, axis=0)
        dxmu1 = dxhat * ivar
        dsqrtvar = -1.0 / (sqrtvar**2) * divar
        dvar = 0.5 * 1.0 / np.sqrt(var + self.eps) * dsqrtvar
        dsq = 1.0 / N * np.ones((N, D)) * dvar
        dxmu2 = 2 * xmu * dsq
        dx1 = (dxmu1 + dxmu2)
        dmu = -1 * np.sum(dxmu1 + dxmu2, axis=0)
        dx2 = 1.0 / N * np.ones((N, D)) * dmu
        dx = dx1 + dx2

        return dx
