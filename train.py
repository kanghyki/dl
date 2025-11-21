from two_layer_net import TwoLayerNet
from optimizer import SGD
from trainer import Trainer
from util import load_data

max_epoch = 30000
batch_size = 30
hidden_size = 10
learning_rate = 0.1

x, t = load_data()
model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
optimizer = SGD(lr=learning_rate)

trainer = Trainer(model, optimizer)
trainer.fit(x, t, max_epoch, batch_size, eval_interval=10)
trainer.plot()
