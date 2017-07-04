import .point as pt
import numpy as np


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


def points_exponential_discontinuous_decline_noisy(y_i, d, x_max, x_disc, y_jump_factor=5.0, x_min=0.0, num=50, noise=0.1, noise_mean=1.0):
    xdata = np.linspace(x_min, x_max)
    ydata = exponential_decline(y_i, d, xdata)
    ydata_noise = ydata * np.random.normal(noise_mean, noise, ydata.shape)
    ydata_noise[np.where(xdata >= x_disc)] = ydata_noise[np.where(xdata >= x_disc)] + y_i/y_jump_factor
    return [pt.Point(x, y) for x, y in zip(xdata, ydata_noise)]
