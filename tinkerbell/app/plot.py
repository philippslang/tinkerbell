import numpy as np
from matplotlib import pyplot as plt

plt.style.use('ggplot')


def render_points(ax, points):
    xcoords = [point.x for point in points]
    ycoords = [point.y for point in points]
    ax.plot(xcoords, ycoords, 'x')


def render_curve(ax, curve, label=None):
    minmax = np.min(curve.t), np.max(curve.t)
    xrender = np.linspace(*minmax, 200)
    ax.plot(xrender, curve(xrender), lw=2, label=label)


def plot_points(pointsiterable):
    """
    Takes a list of lists of points.
    """
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    for points in pointsiterable:
        render_points(ax, points)
    plt.show()


def plot_curves(curvessiterable, labels=None):
    """
    Takes a list of curves.
    """
    make_legend = True
    if not labels:
        labels = [None] * len(curvessiterable)
        make_legend = False
    fig = plt.figure() 
    ax = fig.add_subplot(111)    
    for curve, label in zip(curvessiterable, labels):
        render_curve(ax, curve, label)
    if make_legend:
        ax.legend()
    plt.show()


def plot_points_curves(pointscurvessiterable):
    """
    Takes list of points-curve tuples.
    """
    fig = plt.figure() 
    ax = fig.add_subplot(111)    
    for points, curve in pointscurvessiterable:
        render_curve(ax, curve)
        render_points(ax, points)
    plt.show()