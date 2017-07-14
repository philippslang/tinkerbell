import numpy as np
from matplotlib import pyplot as plt
import logging as log

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


STYLEFALLBACK = {'linestyle': 'solid', 'linewidth': 2, 'alpha': 0.7}
appstyletomplstyle = {'p': {'marker': 'x', 'linestyle': 'None'}, 'l': STYLEFALLBACK}


def plot(xyarraytuplesiterable, styles=[], labels=[], show=True):
    if not styles:
        styles = ['p'] * len(xyarraytuplesiterable)
    legend = True
    if not labels:
        legend = False
        labels = [None] * len(xyarraytuplesiterable)
    fig = plt.figure() 
    ax = fig.add_subplot(111)  
    i = 0  
    for (x, y), style, label in zip(xyarraytuplesiterable, styles, labels):
        try:
            plotargs = appstyletomplstyle[style]
        except:            
            series = label
            if not series:
                series = str(i)
            log.warn('Plot style \'{0}\' for series \'{1}\' not recognized, using fallback.'.format(style, label))
            plotargs = STYLEFALLBACK       
        ax.plot(x, y, **plotargs, label=label)
        i = i+1
    if legend:
        plt.legend()
    if show:
        plt.show()