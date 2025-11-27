from two_layer_net import TwoLayerNet
from optimizer import SGD, AdaGrad, Adam, Momentum
from trainer import Trainer
from util import load_data

max_epoch = 100
batch_size = 100
hidden_size = 50

x, t = load_data()
breakpoint()
model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
optimizer = Adam(lr=0.001)

trainer = Trainer(model, optimizer)
trainer.fit(x, t, max_epoch, batch_size)
trainer.plot()
