import point as pt
import factory as fc
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
from scipy.interpolate import splrep

COLUMNS = ['x_disc', 't0', 't1', 't2', 't3', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6']
OFFSET_T = 1
OFFSET_C = 5
NUM_C = 7
NUM_T = 4
Y_MAX = 50.0
K = 2
FNAME = 'data_demo/discontinuous00.csv'

if __name__ == '__main__':
    y_i = Y_MAX
    d = 0.1
    x_max = y_i
    k = K
    dx = x_max/50
    realizations_per_x_disc = 20
    num_x_discs = 20
    np.random.seed(42)

    data = pd.DataFrame(0.0, index=np.arange(num_x_discs*realizations_per_x_disc), columns=COLUMNS)

    irow = 0
    for x_disc in np.linspace(10.0, 35.0, num_x_discs):
        for irealization in range(realizations_per_x_disc):
            # disturbance time
            data.iat[irow, 0] = x_disc

            pts = fc.points_exponential_discontinuous_decline_noisy(y_i, d, x_max, x_disc)
            pts_coords = [np.array(pt.point_coordinates(pts, i)) for i in range(2)]
            
            xmax = pts_coords[0].max()
            # only internal knots needed for splrep()
            t = [x_disc-dx, x_disc, x_disc+dx, xmax-(xmax-x_disc+dx)/3]
            # and we only save the interior ones, the other ones can be deferred from order
            data.loc[irow, OFFSET_T:OFFSET_T+NUM_T] = t[:]

            t, c, k = splrep(*pts_coords, k=k, task=-1, t=t)  
    
            data.loc[irow, OFFSET_C:OFFSET_C+NUM_C] = c[:-3]  

            irow += 1

    data.to_csv(FNAME, index=False)





        

    

