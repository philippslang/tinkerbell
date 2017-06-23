"""
.. module:: factories
    :platform: Unix, Windows
    :synopsis: Facilitates creation of curves

.. moduleauthor:: Philipp Lang

"""

import json
import numpy as np
from nurbs import Curve as nc
from nurbs import utilities as ncutils
from matplotlib import pyplot as plt

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def read_points(fname):
    with open(fname) as f:
        data = json.load(f)
    coords = data['coordinates']
    xcoords = coords['x']
    ycoords = coords['y']
    return [Point(x, y) for x, y in zip(xcoords, ycoords)]

def plot_points(ax, points):
    xcoords = [point.x for point in points]
    ycoords = [point.y for point in points]
    ax.plot(xcoords, ycoords, 'x')

def least_square(points, h, p=3):
    """
    Generates a curve that approximates points. The curve contains the first and last data
    points and approximates the data polygon in the sense of least square.

    :param points: input points to approximate
    :type points: Point
    :param h: input number of control points
    :type points: int
    :param p: input degree of curve
    :type p: float
    :return: nurbs.Curve
    """
    # D[0..n] = C[0..n] n+1 points to approximat4
    n = len(points)-1
    # P[0..h] h+1 control points, incl. first and last (which equal D[0] and D[n])
    h = h-1
    if not n > h >= p >= 1:
        raise ValueError('Inputs dont honor len(points)-1 > h-1 >= p >= 1')
    pass


points = read_points('data_demo/points_00.json')
fig = plt.figure() 
ax = fig.add_subplot(111)
plot_points(ax, points)
plt.show()



        

    

