import numpy as np
import tinkerbell.domain.point as tbdpt
import tinkerbell.domain.make as tbdmk


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


def points_exponential_discontinuous_declinelinear_noisy(yi, d, xmax, xdisc, y_jumpfactor=5.0, num=50, noise=0.1, noise_mean=1.0):
    xmin = 0.0
    xdata = np.linspace(xmin, xmax, num)
    ixdisc = np.where(xdata > xdisc)    
    ydata = exponential_decline(yi, d, xdata)
    ydata_noise = ydata * np.random.normal(noise_mean, noise, ydata.shape)    
    try:
        ixdisc = ixdisc[0][0]       
        xdata_disc = np.linspace(xmin, xmax-xdisc, len(ydata_noise[ixdisc:]))
        ydata = exponential_decline(ydata_noise[ixdisc] + yi/y_jumpfactor, d, xdata_disc)
        ydata = ydata * np.random.normal(noise_mean, noise, ydata.shape)
        ydata_noise[ixdisc:] = ydata[:]
    except:
        ixdisc = None
    return [tbdpt.Point(x, y) for x, y in zip(xdata, ydata_noise)], ixdisc


def points_exponential_discontinuous_declinebase2_noisy(yi, d, pmax, xdisc, y_jumpfactor=5.0, num=50, noise=0.1, noise_mean=1.0):
    pmin = 0.1
    xdata = np.logspace(pmin, pmax, num, base=2.0)
    ixdisc = np.where(xdata > xdisc)    
    ydata = exponential_decline(yi, d, xdata)
    ydata_noise = ydata * np.random.normal(noise_mean, noise, ydata.shape)    
    try:
        ixdisc = ixdisc[0][0]       
        xdata_disc = xdata[ixdisc:] - xdisc
        ydata = exponential_decline(ydata_noise[ixdisc] + yi/y_jumpfactor, d, xdata_disc)
        ydata = ydata * np.random.normal(noise_mean, noise, ydata.shape)
        ydata_noise[ixdisc:] = ydata[:]
    except:
        ixdisc = None
    return [tbdpt.Point(x, y) for x, y in zip(xdata, ydata_noise)], ixdisc


def knots_internal_four_heavy_right(xcenter, xmax, dx):
    return [xcenter-dx, xcenter, xcenter+dx, xmax-(xmax-xcenter+dx)/3]


