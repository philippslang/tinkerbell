import numpy as np
import tinkerbell.app.plot as tbapl
from matplotlib import pyplot as plt

import example_generate_time_stage_training_set as exgen
import example_regress_on_time_stage_training_set as exregr
import example_load_time_stage_lstm_predict as expred


FNAME_IMG = 'img/time_stage_lstm.png'


def xy_from_npy(fname):
    data = np.load(fname)
    return (data[0], data[1])

xlim = (-5, expred.XMAX*1.05)
ylim = (-2, exgen.Y0*1.1)
lims = (xlim, ylim)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')

training_xy = xy_from_npy(exregr.FNAME_TRAIN)
inputxdisc_xy = ((exgen.XDISC), (0))
tbapl.render(ax1, [training_xy, inputxdisc_xy], styles=['p', 'ix'], lim=lims)
ax1.set_title('Training')

inputy0_xy = ((0), (expred.Y0))
inputxdisc_xy = ((expred.XDISC), (0))
#tbapl.render(ax2, [training_xy, inputy0_xy, inputxdisc_xy], styles=['p', 'iy', 'ix'], lim=lims)
tbapl.render(ax2, [inputy0_xy, inputxdisc_xy], styles=['iy', 'ix'], lim=lims)
ax2.set_title('Query')

prediction_xy = xy_from_npy(expred.FNAME_PRED)
#tbapl.render(ax3, [training_xy, inputy0_xy, inputxdisc_xy, prediction_xy], styles=['p', 'iy', 'ix', 'l'], lim=lims)
tbapl.render(ax3, [inputy0_xy, inputxdisc_xy, prediction_xy], styles=['iy', 'ix', 'l'], lim=lims)
ax3.set_title('Prediction')

blind_xy = xy_from_npy(expred.FNAME_BLIND)
tbapl.render(ax4, [prediction_xy, blind_xy], styles=['l', 'p'], lim=lims)
ax4.set_title('Prediction vs. reference')

fig.set_size_inches(14, 9)
plt.savefig(FNAME_IMG, dpi=300)
plt.show()


