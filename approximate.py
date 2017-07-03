import point as pt
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import make_lsq_spline, BSpline, UnivariateSpline, splrep
from scipy.optimize import fsolve

if 0:
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

else:
    pts_fname = 'data_demo/points_02.json'
    pts = pt.read_points(pts_fname)
    pts_coords = [np.array(pt.point_coordinates(pts, i)) for i in range(2)]

    def fit(sfactor):
        return  splrep(*pts_coords, k=3, s=len(pts)*sfactor)

    for sfactor in [1, 2, 3, 4]:
        t, c, k = fit(sfactor)
        spl = BSpline(t, c, k)
        print(len(spl.t))

#print(spl.t)
#print(spl.c)
#print(len(spl.t))


fig = plt.figure() 
ax = fig.add_subplot(111)
pt.plot_points(ax, pts)
xcoords_plot = np.linspace(pts_coords[0].min(), pts_coords[0].max(), 200)
ax.plot(xcoords_plot, spl(xcoords_plot), 'g-', lw=3, label='LSQ spline')
plt.show()


        

    

