import numpy as np
import tinkerbell.app.plot as tbapl
from matplotlib import pyplot as plt

import example_generate_time_stage_training_set as exgen
import example_regress_on_time_stage_training_set as exregr
import example_load_time_stage_lstm_predict as expred

def xy_from_npy(fname):
    data = np.load(fname)
    return (data[0], data[1])

xlim = (-5, expred.XMAX*1.05)
ylim = (-2, exgen.Y0*1.1)
lims = (xlim, ylim)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')

training_xy = xy_from_npy(exregr.FNAME_TRAIN)
tbapl.render(ax1, [training_xy], lim=lims)
ax1.set_title('Training')

inputy0_xy = ((0), (expred.Y0))
inputxdisc_xy = ((expred.XDISC), (0))
tbapl.render(ax2, [training_xy, inputy0_xy, inputxdisc_xy], styles=['p', 'iy', 'ix'], lim=lims)
ax2.set_title('Input')

prediction_xy = xy_from_npy(expred.FNAME_PRED)
tbapl.render(ax3, [training_xy, inputy0_xy, inputxdisc_xy, prediction_xy], styles=['p', 'iy', 'ix', 'l'], lim=lims)
ax3.set_title('Prediction')

blind_xy = xy_from_npy(expred.FNAME_BLIND)
tbapl.render(ax4, [blind_xy, prediction_xy], styles=['p', 'l'], lim=lims)
ax4.set_title('Prediction vs. reference')

plt.show()


