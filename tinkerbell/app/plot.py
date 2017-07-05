import numpy as np
from matplotlib import pyplot as plt


def render_points(ax, points):
    xcoords = [point.x for point in points]
    ycoords = [point.y for point in points]
    ax.plot(xcoords, ycoords, 'x')


def render_curve(ax, curve):
    minmax = np.min(curve.t), np.max(curve.t)
    xrender = np.linspace(*minmax, 200)
    ax.plot(xrender, curve(xrender), lw=2)


def plot_points(pointsiterable):
    """
    Takes a list of lists of points.
    """
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    for points in pointsiterable:
        render_points(ax, points)
    plt.show()


def plot_curves(curvessiterable):
    """
    Takes a list of curves.
    """
    fig = plt.figure() 
    ax = fig.add_subplot(111)    
    for curve in curvessiterable:
        render_curve(ax, curve)
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