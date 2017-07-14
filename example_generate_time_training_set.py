import tinkerbell.domain.point as tbdpt
import tinkerbell.app.make as tbamk
import tinkerbell.app.rcparams as tbarc
import tinkerbell.app.plot as tbapl
import pandas as pd
import numpy as np


y0 = tbarc.rcparams['shale.exp.y0_mean']
d = 0.1
xmax = 80.0
num_points = 100
xdisc = 20.0

columns = ('x', 'y')
data = pd.DataFrame(0.0, index=np.arange(num_points), columns=columns)

np.random.seed(42)
pts, _ = tbamk.points_exponential_discontinuous_declinelinear_noisy(y0, d, xmax, xdisc, num=num_points)

xycoords = tbdpt.point_coordinates(pts)

tbapl.plot([xycoords])

data['x'] = xycoords[0]
data['y'] = xycoords[1]
data.to_csv(tbarc.rcparams['shale.exp.csvtimefname'], index=False)
