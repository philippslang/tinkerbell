import tinkerbell.app.plot as tbapl
import tinkerbell.domain.make as tbdmk
import tinkerbell.domain.curve as tbdcv
import tinkerbell.app.make as tbamk
import tinkerbell.app.rcparams as tbarc
import pandas as pd
import numpy as np


num_xdisc = 20
num_realizations = 20
xdiscspace = (10.0, 35.0)
y0_mean = tbarc.rcparams['shale.exp.y0_mean']
y0_stddev = 5.0
d = 0.1
k = tbarc.rcparams['shale.exp.k']
xmax = y0_mean
dx = tbarc.rcparams['shale.exp.dx']

num_knots = tbdmk.num_knots_curve_lsq(k, tbarc.rcparams['shale.exp.num_knots_internal'])
columns = ('y0', 'xdisc', *tbdcv.flat_header(num_knots, num_knots))
data = pd.DataFrame(0.0, index=np.arange(num_xdisc*num_realizations), columns=columns)

idatarow = 0
np.random.seed(42)
for xdisc in np.linspace(*xdiscspace, num_xdisc):
    for irealization in range(num_realizations):
        y0 = y0_mean
        pts = tbamk.points_exponential_discontinuous_decline_noisy(y0, d, xmax, xdisc)
        t = tbamk.knots_internal_four_heavy_right(xdisc, xmax, dx)
        crv = tbdmk.curve_lsq_fixed_knots(pts, t, k)

        data.iat[idatarow, 0] = y0
        data.iat[idatarow, 1] = xdisc
        data.loc[idatarow, 2:] = crv.to_flat() 
        idatarow += 1

data.to_csv(tbarc.rcparams['shale.exp.csvfname'], index=False)