import time
import numpy as np
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.current_epoch = 0

    def fit(self, x, t, max_epoch=10, batch_size=32, checkpoint=True):
        data_size = len(x)
        max_iters = data_size // batch_size

        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        cp_loss = np.inf
        cp_params = []

        start_time = time.time()
        for _ in range(max_epoch):
            # 뒤섞기
            idx = np.random.permutation(np.arange(data_size))
            x = x[idx]
            t = t[idx]

            for iters in range(max_iters):
                batch_x = x[iters*batch_size:(iters+1)*batch_size]
                batch_t = t[iters*batch_size:(iters+1)*batch_size]

                loss = model.forward(batch_x, batch_t)
                model.backward()
                optimizer.update(model.params, model.grads)

                total_loss += loss
                loss_count += 1

                if iters == max_iters - 1:
                    avg_loss = total_loss / loss_count
                    elapsed_time = time.time() - start_time
                    print(f'| epoch: {self.current_epoch + 1:5} |  iter:{iters+1:>5} / {max_iters:>5} | time:{elapsed_time:5.1f}s | avg loss:{avg_loss:10.8f} |')
                    self.loss_list.append(float(avg_loss))

                    if avg_loss < cp_loss:
                        print("checkpoint")
                        cp_loss = avg_loss
                        cp_params = self.model.save_params()

                    total_loss, loss_count = 0, 0
            self.current_epoch += 1

        if checkpoint:
            print(cp_loss)
            self.model.load_params(cp_params)
            plot_decision_boundary(model, x, t)

    def plot(self, ylim=None):
        x = np.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='train')
        plt.xlabel('Iter')
        plt.ylabel('Loss')
        plt.show()

def plot_decision_boundary(model, x, t):
    if t.ndim == 2:
        t_classes = np.argmax(t, axis=1)
    else:
        t_classes = t

    h = 0.02
    x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
    y_min, y_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    X_grid = np.c_[xx.ravel(), yy.ravel()]
    predictions = model.predict(X_grid, train_mode=False)
    Z = np.argmax(predictions, axis=1)
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(12, 10))
    plt.contourf(xx, yy, Z, alpha=0.4, levels=2, cmap='RdYlBu')
    
    colors = ['red', 'yellow', 'blue']
    markers = ['o', 's', '^']
    for i in range(3):
        idx = t_classes == i
        plt.scatter(x[idx, 0], x[idx, 1], 
                   c=colors[i], marker=markers[i], 
                   s=80, edgecolors='black', linewidths=1.5,
                   label=f'Class {i}', alpha=0.8)
    
    plt.xlabel('X1', fontsize=12)
    plt.ylabel('X2', fontsize=12)
    plt.title('Decision Boundary', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.show()
