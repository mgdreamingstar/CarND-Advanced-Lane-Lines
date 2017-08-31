import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-10,10,100)
y = np.dot(x,2)
y = np.exp(x)
p = np.polyfit(x,y,2)
pf = np.poly1d(p)

plt.plot(x,pf(x))
