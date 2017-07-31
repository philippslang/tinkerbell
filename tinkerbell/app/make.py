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
        ydata = exponential_decline(ydata_noise[ixdisc] + (yi/y_jumpfactor) * np.random.normal(noise_mean, noise*3), d, xdata_disc)
        ydata = ydata * np.random.normal(noise_mean, noise*2, ydata.shape)
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
    return [xcenter-dx, xcenter, xcenter+dx, xmax-(xmax-xcenter+dx)/2]


def detect_stages(x, y, stage_zero=0, num_stages_max=None, num_samples_window=2):
    """
    Returns stage vector.
    """
    stages = np.empty_like(x, dtype=np.int16)
    stages.fill(stage_zero)
    num_stage_current = 1  
    for i in range(num_samples_window, len(stages)):
        if y[i] > np.max(y[i-num_samples_window:i]):
            stages[i:] = stages[i:]+1
            num_stage_current += 1
            if num_stages_max is not None and num_stage_current >= num_stages_max:
                break
    return stages


def test_stage_detection():
    pts, ixdisc = points_exponential_discontinuous_declinelinear_noisy(100.0, 0.1, 100.0, 50.0)
    stages = detect_stages(*tbdpt.point_coordinates(pts))
    stage_changes = stages[1:] != stages[:-1]
    stage_change_idcs = np.where(stage_changes==True)
    assert stage_change_idcs[0][0]+1 == ixdisc

