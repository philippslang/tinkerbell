import point as pt
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import make_lsq_spline, BSpline


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
    # D[0..n] = C[0..n] n+1 points to approximate
    n = len(points)-1
    # P[0..h] h+1 control points, incl. first and last (which equal D[0] and D[n])
    h = h-1
    if not n > h >= p >= 1:
        raise ValueError('Inputs dont honor len(points)-1 > h-1 >= p >= 1')
    pass

if 0:
    pts_fname = 'data_demo/points_01.json'
    pts = pt.read_points(pts_fname)
    pts_coords = [np.array(pt.point_coordinates(pts, i)) for i in range(2)]

    # approcimating spline
    t = [10, 25, 40]
    t = [25]
    t = [np.median(pts_coords[0])]
else:
    pts_fname = 'data_demo/points_02.json'
    pts = pt.read_points(pts_fname)
    pts_coords = [np.array(pt.point_coordinates(pts, i)) for i in range(2)]

    # approcimating spline
    t = np.linspace(10, 30, 7)

k = 3
t = np.r_[(pts_coords[0][0],)*(k+1), t, (pts_coords[0][-1],)*(k+1)]
spl = make_lsq_spline(*pts_coords, t, k)

print(spl.t)
print(spl.c)
print(spl.k)

fig = plt.figure() 
ax = fig.add_subplot(111)
pt.plot_points(ax, pts)
ax.plot(pts_coords[0], spl(pts_coords[0]), 'g-', lw=3, label='LSQ spline')
plt.show()


        

    

