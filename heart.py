import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.animation import FFMpegWriter


x = np.linspace(-1.816, 1.816, 1000)
a_set = np.arange(0, 46, 0.1)
y = np.zeros(x.shape[0])

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.ion()
plt.show()
for a in a_set:
    try:
        ax.lines.remove(lines[0])
    except Exception:
        pass
    for i in range(x.shape[0]):
        s1 = np.abs(x[i]) ** (2/3)
        s2 = 0.9*(3.3 - x[i] ** 2) ** (1/2)
        s3 = math.sin(a * math.pi * x[i] + 3)
        y[i] = s1 + s2 * s3

    lines = plt.plot(x, y, color='red', linewidth=2.0)
    plt.xlim(-3, 3)
    plt.ylim(-2.5, 2.5)
    plt.pause(0.03) 
    


"""

fig, ax = plt.subplots()
x, y = [], []
line, = ax.plot([], [], color='red', linewidth=2.0, animated=False)
func = np.abs(x[i]) ** (2/3) + 0.9*(3.3 - x[i] ** 2) ** (1/2) * math.sin(a * math.pi * x[i] + 3)

def init():
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2.5, 2.5)
    return line,

def updata(n):
    x.append(n)
    y.append()
    
"""