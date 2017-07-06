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


def points_exponential_discontinuous_decline_noisy(yi, d, xmax, xdisc, y_jumpfactor=5.0, xmin=0.0, num=50, noise=0.1, noise_mean=1.0):
    xdata = np.linspace(xmin, xmax)
    ydata = exponential_decline(yi, d, xdata)
    ydata_noise = ydata * np.random.normal(noise_mean, noise, ydata.shape)
    ydata_noise[np.where(xdata >= xdisc)] = ydata_noise[np.where(xdata >= xdisc)] + yi/y_jumpfactor
    return [pt.Point(x, y) for x, y in zip(xdata, ydata_noise)]


def knots_four_heavy_right(xcenter, xmax, dx):
    """
    Internal knots.
    """
    return [xcenter-dx, xcenter, xcenter+dx, xmax-(xmax-xcenter+dx)/3]


def curve_lsq_fixed_knots(points, t, k):
    """
    Points, internal knots and order.
    """
    points_xy = [pt.point_coordinates(points, i) for i in range(2)]
    tck = spint.splrep(*points_xy, k=k, task=-1, t=t)
    return cv.Curve(*tck)