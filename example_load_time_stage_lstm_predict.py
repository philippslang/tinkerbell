"""
Recurrent neural network model

TODO
experiment with lag (diff n)
experiment with number of features (differences of 2, 4, 6 last observations; 
  these would also have to include the stage value)
ditto number neurons (> 3)
ditto number of layers, I'd expect one needed per disc
moving average gradient
optimizing for the error metric (mean absolute error) instead of RMSE
change timesteps http://machinelearningmastery.com/use-timesteps-lstm-networks-time-series-forecasting/
update model http://machinelearningmastery.com/update-lstm-networks-training-time-series-forecasting/
"""

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import sklearn.preprocessing as skprep
import sklearn.metrics as skmet
from sys import exit
import tinkerbell.app.make as tbamk
import tinkerbell.app.rcparams as tbarc
import tinkerbell.app.plot as tbapl
import tinkerbell.domain.point as tbdpt
import keras.models as kem
import example_regress_on_time_stage_training_set
import example_generate_time_stage_training_set


def do_the_thing():
  y0 = tbarc.rcparams['shale.exp.y0_mean']
  d = example_generate_time_stage_training_set.D
  xmax = example_generate_time_stage_training_set.XMAX
  num_points = example_generate_time_stage_training_set.NUM_POINTS
  xdisc = example_generate_time_stage_training_set.XDISC

  np.random.seed(42)
  pts, ixdisc = tbamk.points_exponential_discontinuous_declinelinear_noisy(y0, d, xmax, xdisc, num=num_points)

  ycomp_pts, xcomp_pts =  tbdpt.point_coordinates(pts)

  tbapl.plot([(xcomp_pts, ycomp_pts)])

  stagedelta = np.zeros_like(ycomp_pts)
  stagedelta[ixdisc] = 1.0

  input = np.transpose(np.array([ycomp_pts, stagedelta])) 
  normalizer_input = skprep.MinMaxScaler(feature_range=(-1, 1))
  input_normalized = normalizer_input.fit_transform(input)
  print(input_normalized.shape)

  ydeltaoutput = np.diff(ycomp_pts)
  ydeltaoutput = ydeltaoutput.reshape(-1, 1)
  normalizer_ydeltaoutput = skprep.MinMaxScaler(feature_range=(-1, 1))
  normalizer_ydeltaoutput.fit(ydeltaoutput)

  fname_model = example_regress_on_time_stage_training_set.FNAME 
  model = kem.load_model(fname_model)

  yhat = [ycomp_pts[0]]
  # didnt use diff (which shortens array by one), so we ommit prediction based on last input to not exceed reference solution length
  for i in range(input_normalized.shape[0]-1): 
      inputi_normalized = np.array([input_normalized[i,:]])
      inputi_normalized = inputi_normalized.reshape(inputi_normalized.shape[0], 1, 
        inputi_normalized.shape[1])
      ydelta_normalized = model.predict(inputi_normalized, batch_size=1)
      ydelta = normalizer_ydeltaoutput.inverse_transform(ydelta_normalized)
      ydelta = ydelta[0, 0]
      yhat += [yhat[-1]+ydelta]

  #tbapl.plot([(xplot, ycomp_pts), (xplot, yhat)], styles=['p', 'l'], labels=['yblind', 'yhat'])
  


if __name__ == '__main__':
  #example_regress_on_time_stage_training_set.do_the_thing(True, 1000, 4)
  do_the_thing()


