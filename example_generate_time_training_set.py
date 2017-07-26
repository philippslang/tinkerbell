import tinkerbell.domain.point as tbdpt
import tinkerbell.app.make as tbamk
import tinkerbell.app.rcparams as tbarc
import tinkerbell.app.plot as tbapl
import pandas as pd
import numpy as np


def do_the_thing():
    y0 = tbarc.rcparams['shale.lstm.y0_mean']
    d = tbarc.rcparams['shale.lstm.d']
    xmax = tbarc.rcparams['shale.lstm.xmax']
    num_points = tbarc.rcparams['shale.lstm.num_points']

    columns = ('x', 'y', 'stage')
    data = pd.DataFrame(0.0, index=np.arange(num_points), columns=columns)

    np.random.seed(42)
    pts, _ = tbamk.points_exponential_discontinuous_declinelinear_noisy(y0, d, xmax, xmax, num=num_points)

    xycoords = tbdpt.point_coordinates(pts)

    tbapl.plot([xycoords])

    data['x'] = xycoords[0]
    data['y'] = xycoords[1]
    data.to_csv(tbarc.rcparams['shale.lstm.fnamecsv'], index=False)


if __name__ == '__main__':
    do_the_thing()
