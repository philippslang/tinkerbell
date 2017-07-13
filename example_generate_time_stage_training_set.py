import tinkerbell.domain.point as tbdpt
import tinkerbell.app.make as tbamk
import tinkerbell.app.rcparams as tbarc
import pandas as pd
import numpy as np


y0 = tbarc.rcparams['shale.exp.y0_mean']
d = 0.05
xmax = 100.0
num_points = 100
xdisc = 40.0
ixdisc = int(xdisc/(xmax/num_points))

columns = ('x', 'stage', 'y')
data = pd.DataFrame(0.0, index=np.arange(num_points), columns=columns)

np.random.seed(42)
pts = tbamk.points_exponential_discontinuous_decline_noisy(y0, d, xmax, xdisc, num=num_points)

data['x'] = tbdpt.point_coordinates(pts, idx=0)
data['y'] = tbdpt.point_coordinates(pts, idx=1)
data['stage'].iloc[ixdisc:] = 1.0
data.to_csv(tbarc.rcparams['shale.exp.csvtimestagefname'], index=False)
