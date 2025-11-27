import numpy as np

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]

class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = []
            for i in range(len(params)):
                self.v.append(np.zeros_like(params[i]))

        for i in range(len(params)):
            self.v[i] = self.momentum * self.v[i] - self.lr * grads[i]
            params[i] += self.v[i]

class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = []
            for i in range(len(params)):
                self.h.append(np.zeros_like(params[i]))

        for i in range(len(params)):
            self.h[i] += grads[i] * grads[i]
            params[i] -= self.lr * grads[i] / (np.sqrt(self.h[i]) + 1e-7)

class Adam:
    def __init__(self, lr=0.001):
        self.lr = lr
        self.b1 = 0.9
        self.b2 = 0.999
        self.init = False
        self.t = 0
        self.m = []
        self.v = []
        self.eps = 1e-8

    def update(self, params, grads):
        breakpoint()
        if self.init is False:
            for i in range(len(params)):
                self.m.append(np.zeros_like(params[i]))
                self.v.append(np.zeros_like(params[i]))
            self.init = True

        self.t += 1
        for i in range(len(params)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * grads[i]
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (grads[i]**2)
            mh = self.m[i] / (1 - self.b1**self.t)
            vh = self.v[i] / (1 - self.b2**self.t)
            params[i] -= self.lr * mh / (np.sqrt(vh) + self.eps)

