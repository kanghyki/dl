import numpy as np
import matplotlib.pyplot as plt

def load_data(seed=1984):
    np.random.seed(seed)
    N = 100  # 클래스당 샘플 수
    DIM = 2  # 데어터 요소 수
    CLS_NUM = 3  # 클래스 수

    x = np.zeros((N*CLS_NUM, DIM))
    t = np.zeros((N*CLS_NUM, CLS_NUM))

    for j in range(CLS_NUM):
        for i in range(N): # N*j, N*(j+1)):
            rate = i / N
            radius = 1.0*rate
            theta = j*4.0 + 4.0*rate + np.random.randn()*0.2

            ix = N*j + i
            x[ix] = np.array([radius*np.sin(theta),
                              radius*np.cos(theta)]).flatten()
            t[ix, j] = 1

    return x, t

def plot_weight_distribution(weights, title, figsize=(10, 6)):
    weights_flat = weights.flatten()

    plt.figure(figsize=figsize)
    plt.hist(weights_flat, bins=50, range=(-1, 1), alpha=0.7, edgecolor='black', color='skyblue')
 
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
