import numpy as np
from numpy.typing import NDArray

# functions

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
        self.params, self.grads= [], []
        self.out: NDArray = None

    def forward(self, x):
        self.out = sigmoid(x)
        return self.out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
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
