import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-10,10,10000)
a = np.array([  1,10,20])
pa = np.poly1d(a)
plt.plot(x,pa(x))
pa(2)
pa(10)
