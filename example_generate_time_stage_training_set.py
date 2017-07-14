import tinkerbell.domain.point as tbdpt
import tinkerbell.app.make as tbamk
import tinkerbell.app.rcparams as tbarc
import tinkerbell.app.plot as tbapl
import pandas as pd
import numpy as np

XMAX = 90.0
XDISC = 40.0
NUM_POINTS = 75
D = 0.05

if __name__ == '__main__':
    y0 = tbarc.rcparams['shale.exp.y0_mean']
    d = D
    xmax = XMAX
    num_points = NUM_POINTS
    xdisc = XDISC

    columns = ('x', 'stage', 'y')
    data = pd.DataFrame(0.0, index=np.arange(num_points), columns=columns)

    np.random.seed(42)
    pts, ixdisc = tbamk.points_exponential_discontinuous_declinelinear_noisy(y0, d, xmax, xdisc, num=num_points)

    xycoords = tbdpt.point_coordinates(pts)

    tbapl.plot([xycoords])

    data['x'] = xycoords[0]
    data['y'] = xycoords[1]
    data['stage'].iloc[ixdisc:] = 1.0
    data.to_csv(tbarc.rcparams['shale.exp.csvtimestagefname'], index=False)
