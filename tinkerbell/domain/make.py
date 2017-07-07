from . import point as pt
from . import curve as cv
import numpy as np
import scipy.interpolate as spint


def curve_lsq_fixed_knots(points, t, k):
    """
    Points, internal knots and order.
    """
    points_xy = [pt.point_coordinates(points, i) for i in range(2)]
    tck = spint.splrep(*points_xy, k=k, task=-1, t=t)
    return cv.Curve(*tck)


def num_knots_curve_lsq(k, num_internal_knots):
    """
    Returns the number of total knots created by curve_lsq_fixed_knots.
    """
    return (k+1)*2+num_internal_knots


def knots_from_internal_knots(k, internal_knots, xmin, xmax):
    order = k+1
    return np.r_[[xmin]*order, internal_knots, [xmax]*order]
