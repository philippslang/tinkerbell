import point as pt
import factory as fc
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import make_lsq_spline, BSpline, UnivariateSpline, splrep
from scipy.optimize import fsolve


y_i = 50
d = 0.1
x_max = 50.0
k = 2

for x_disc in np.linspace(10.0, 30.0, 8):
    pts = fc.points_exponential_discontinuous_decline_noisy(y_i, d, x_max, x_disc)
    pts_coords = [np.array(pt.point_coordinates(pts, i)) for i in range(2)]

    if 0:
        sfactor = 5
        t, c, k = splrep(*pts_coords, k=k, s=len(pts)*sfactor)
    else:
        dx = 4.0
        xmax = pts_coords[0].max()
        # only internal knots needed for splrep()
        t = [x_disc-dx, x_disc, x_disc+dx]
        t = [x_disc-dx, x_disc, x_disc+dx, xmax-(xmax-x_disc+dx)/3]
        #t = [x_disc-dx, x_disc, x_disc+dx, x_disc+dx*2.0]
        #t = [x_disc-dx, x_disc, x_disc+dx/2, x_disc+dx]
        t, c, k = splrep(*pts_coords, k=k, task=-1, t=t)        

    spl = BSpline(t, c, k)

    fig = plt.figure() 
    ax = fig.add_subplot(111)
    pt.plot_points(ax, pts)
    xcoords_plot = np.linspace(pts_coords[0].min(), pts_coords[0].max(), 200)
    ax.plot(xcoords_plot, spl(xcoords_plot), 'g-', lw=3)
    #print(spl.t)
    print(spl.c)
    plt.show()

#print(len(spl.t))
#print(spl.t)
#print(spl.c)
#print(len(spl.t))





        

    

