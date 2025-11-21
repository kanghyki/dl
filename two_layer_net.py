import numpy as np
from common import Affine, Sigmoid, SoftMaxWithCrossEntropyError
from init_weight import init_xavier

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        # W1 = np.random.randn(I, H)
        W1 = init_xavier(I, H)
        b1 = np.random.randn(H)
        # W2 = np.random.randn(H, O)
        W2 = init_xavier(H, O)
        b2 = np.random.randn(O)

        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2),
        ]

        self.loss_layer = SoftMaxWithCrossEntropyError()

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x, t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
