import point as pt
import numpy as np
from matplotlib import pyplot as plt

def exponential_decline(y_i, d, x):
    """
    Parameters
    ----------
    y_i: float
         Start value.
    d: float
       Decline rate (positive).
    x: float
       Independent variable.
    """
    return y_i*np.exp(-d*x)


y_i = 50
d = 0.1
xdata = np.linspace(0, 50,)
ydata = exponential_decline(y_i, d, xdata) + 10.0
ydata = ydata * np.random.normal(1.0, 0.1, ydata.shape)
pts = [pt.Point(x, y) for x, y in zip(xdata, ydata)]
pt.write_points('data_demo/points_01.json', pts)

fig = plt.figure() 
ax = fig.add_subplot(111)
pt.plot_points(ax, pts)
plt.show()



        

    

