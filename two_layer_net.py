import numpy as np
from common import Affine, BatchNorm, Relu, SoftMaxWithCrossEntropyError

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        W1 = np.random.randn(I, H)
        b1 = np.zeros(H)
        Wend = np.random.randn(H, O) * 0.01
        bend = np.zeros(O)

        self.layers = [
            Affine(W1, b1),
            BatchNorm(H),
            Relu(),
        ]
        for _ in range(15):
            self.layers.append(Affine(np.random.randn(H, H) * 0.01, np.zeros(H)))
            self.layers.append(BatchNorm(H))
            self.layers.append(Relu())

        self.layers.append(Affine(Wend, bend))

        self.loss_layer = SoftMaxWithCrossEntropyError()

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, x, train_mode=True):
        for layer in self.layers:
            layer.train_mode = train_mode
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

    def save_params(self):
        saved = []
        for layer in self.layers:
            layer_params = [np.copy(param) for param in layer.params]
            saved.append(layer_params)
        return saved
    
    def load_params(self, saved_params):
        for layer, saved_layer in zip(self.layers, saved_params):
            for param, saved_param in zip(layer.params, saved_layer):
                param[...] = saved_param
