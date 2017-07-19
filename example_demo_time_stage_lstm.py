import numpy as np
import tinkerbell.app.plot as tbapl
import tinkerbell.app.rcparams as tbarc
from matplotlib import pyplot as plt


fname_img = 'img/time_stage_lstm.png'

def xy_from_npy(fname):
    data = np.load(fname)
    return (data[0], data[1])

xmax = tbarc.rcparams['shale.lstm_stage.xmax']
y0 = tbarc.rcparams['shale.lstm.y0_mean']
xdisc_train = tbarc.rcparams['shale.lstm_stage.xdisc_mean']
xdisc_pred = xdisc_train + xmax*tbarc.rcparams['shale.lstm_stage.xdisc_mult']
y0_pred = tbarc.rcparams['shale.lstm.y0_mean']*tbarc.rcparams['shale.lstm.y0_mult']
xlim = (-5, xmax*1.025)
ylim = (-2, y0*1.1)
lims = (xlim, ylim)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')

training_xy = xy_from_npy(tbarc.rcparams['shale.lstm_stage.fnamentraindata'])
inputxdisc_xy = ((tbarc.rcparams['shale.lstm_stage.xdisc_mean']), (0))
tbapl.render(ax1, [training_xy, inputxdisc_xy], styles=['p', 'ix'], lim=lims)
ax1.set_title('Training')

inputy0_xy = ((0), (y0_pred))
inputxdisc_xy = ((xdisc_pred), (0))
tbapl.render(ax2, [inputy0_xy, inputxdisc_xy], styles=['iy', 'ix'], lim=lims)
ax2.set_title('Query')

prediction_xy = xy_from_npy(tbarc.rcparams['shale.lstm_stage.fnamenpreddata'])
tbapl.render(ax3, [inputy0_xy, inputxdisc_xy, prediction_xy], styles=['iy', 'ix', 'l'], lim=lims)
ax3.set_title('Prediction')

blind_xy = xy_from_npy(tbarc.rcparams['shale.lstm_stage.fnamenblinddata'])
tbapl.render(ax4, [prediction_xy, blind_xy], styles=['l', 'p'], lim=lims)
ax4.set_title('Prediction vs. reference')

fig.set_size_inches(14, 9)
plt.savefig(fname_img, dpi=300)
plt.show()


