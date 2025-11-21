import numpy as np

def init_norm_distribution(prev, next, std_dev = 1.0):
    return np.random.randn(prev, next) * std_dev

def init_xavier(prev, next):
    return np.random.randn(prev, next) * np.sqrt(1 / (prev + next))

def init_he(prev, next):
    return np.random.randn(prev, next) * np.sqrt(2 / prev)
