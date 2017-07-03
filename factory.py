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


def points_exponential_decline_noisy(y_i, d, x_max, x_min=0.0, num=50, noise=0.1):
    xdata = np.linspace(x_min, x_max)
    ydata = exponential_decline(y_i, d, xdata) + 10.0
    ydata_noise = ydata * np.random.normal(1.0, noise, ydata.shape)
    return [pt.Point(x, y) for x, y in zip(xdata, ydata_noise)]


def points_exponential_discontinuous_decline_noisy(y_i, d, x_max, x_disc, y_jump_factor=5.0, x_min=0.0, num=50, noise=0.1):
    xdata = np.linspace(x_min, x_max)
    ydata = exponential_decline(y_i, d, xdata) + y_i/y_jump_factor * np.random.normal(1.0, noise)
    ydata_noise = ydata * np.random.normal(1.0, noise, ydata.shape)
    ydata_noise[np.where(xdata >= x_disc)] = ydata_noise[np.where(xdata >= x_disc)] + 10.0
    return [pt.Point(x, y) for x, y in zip(xdata, ydata_noise)]


if __name__ == '__main__':
    y_i = 50
    d = 0.1
    x_max = 50.0
    pts_exp_cont = points_exponential_decline_noisy(y_i, d, x_max)
    pt.write_points('data_demo/points_01.json', pts_exp_cont)

    x_disc = 20.0
    pts_exp_discont = points_exponential_discontinuous_decline_noisy(y_i, d, x_max, x_disc)
    pt.write_points('data_demo/points_02.json', pts_exp_discont)

    fig = plt.figure() 
    ax = fig.add_subplot(111)
    pt.plot_points(ax, pts_exp_cont)
    pt.plot_points(ax, pts_exp_discont)
    plt.show()



        

    

