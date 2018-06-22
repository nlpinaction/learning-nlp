import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-1, 1, 50)
y = 1.0/(1.0+np.exp(-5*x))
plt.figure()
plt.plot(x, y)
plt.show()