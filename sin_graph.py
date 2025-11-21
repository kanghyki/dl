import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 6, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)


plt.plot(x, y_sin, label='sin')
plt.plot(x, y_cos, label='cos')
plt.xlabel('x')
plt.ylabel('y')
plt.title("sin & cos")

plt.legend()
plt.show()
