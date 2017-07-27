import tinkerbell.domain.point as tbdpt
import tinkerbell.app.make as tbamk
import tinkerbell.app.rcparams as tbarc
import tinkerbell.app.plot as tbapl
import pandas as pd
import numpy as np


if __name__ == '__main__':
    y0 = tbarc.rcparams['shale.lstm.y0_mean']
    d = tbarc.rcparams['shale.lstm_stage.d']
    xmax = tbarc.rcparams['shale.lstm_stage.xmax']
    num_points = tbarc.rcparams['shale.lstm_stage.num_points']
    xdisc = tbarc.rcparams['shale.lstm_stage.xdisc_mean']

    columns = ('x', 'stage', 'y')
    data = pd.DataFrame(0.0, index=np.arange(num_points), columns=columns)

    np.random.seed(42)
    pts, ixdisc = tbamk.points_exponential_discontinuous_declinelinear_noisy(y0, d, xmax, xdisc, num=num_points)

    xycoords = tbdpt.point_coordinates(pts)    

    data['x'] = xycoords[0]
    data['y'] = xycoords[1]
    data['stage'].iloc[ixdisc:] = 1.0
    data.to_csv(tbarc.rcparams['shale.lstm_stage.fnamecsv'], index=False)

    xstage = data['x'].values
    ystage = data['stage'].values
    tbapl.plot([xycoords], labels=['production'], save_as='img/exp_time_two_stages.png', 
      secxyarraytuplesiterable=[(xstage, ystage)], seclabels=['stage'], secstyles=['lstage'],
      hide_labels=True, xlabel='time', ylabel='production', secylabel='stage')
