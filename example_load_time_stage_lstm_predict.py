import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import sklearn.preprocessing as skprep
import sklearn.metrics as skmet
from sys import exit
import tinkerbell.app.make as tbamk
import tinkerbell.app.rcparams as tbarc
import tinkerbell.app.plot as tbapl
import tinkerbell.app.model as tbamd
import tinkerbell.domain.point as tbdpt
import example_regress_on_time_stage_training_set
import example_generate_time_stage_training_set
import logging as log
import pickle

FNAME_BLIND = 'data_demo/time_stage_lstm_blind.npy'
FNAME_PRED = 'data_demo/time_stage_lstm_pred.npy'

XMAX = example_generate_time_stage_training_set.XMAX
Y0 = tbarc.rcparams['shale.exp.y0_mean']*0.7
XDISC = example_generate_time_stage_training_set.XDISC + XMAX*0.15

def do_the_thing():
  y0 = Y0
  d = example_generate_time_stage_training_set.D
  xmax = XMAX
  num_points = example_generate_time_stage_training_set.NUM_POINTS
  xdisc = XDISC

  np.random.seed(42)
  pts, ixdisc = tbamk.points_exponential_discontinuous_declinelinear_noisy(y0, d, xmax, xdisc, num=num_points)

  xcomp_pts, ycomp_pts =  tbdpt.point_coordinates(pts)

  stage = np.zeros_like(ycomp_pts)
  stage[ixdisc:] = 1.0

  features = tbamd.Features(ycomp_pts, stage)
  targets = tbamd.Targets(ycomp_pts)

  fname_normalizer = example_regress_on_time_stage_training_set.FNAME_NORM
  normalizer = tbamd.Normalizer.load(fname_normalizer)

  fname_model = example_regress_on_time_stage_training_set.FNAME 
  model = tbamd.load(fname_model)

  yhat = [ycomp_pts[0]]
  for i in range(1, len(ycomp_pts)-1): 
    # input is first value, last discarded internally due to grad calc
    ylasttwo = [yhat[-1], 0.0]
    stagelasttwo = stage[i-1:i+1]
    features_predict = tbamd.Features(ylasttwo, stagelasttwo)
    features_predict_normalized = normalizer.normalize_features(features_predict)
    features_predict_normalized = features_predict_normalized.reshape(features_predict_normalized.shape[0], 
      1, features_predict_normalized.shape[1])
    target_predicted_normalized = model.predict(features_predict_normalized, batch_size=1)
    target_predicted = normalizer.denormalize_targets(target_predicted_normalized)
    target_predicted = target_predicted[0, 0]       
    yhat += [yhat[-1]+target_predicted]

  input_xy = ((0, y0), (xdisc, 0))
  #print(input_xy)
  xplot = xcomp_pts[:-1]
  yref = ycomp_pts[:-1]
  np.save(FNAME_BLIND, np.array([xplot, yref]))
  np.save(FNAME_PRED, np.array([xplot, yhat]))
  tbapl.plot([(xplot, yref), (xplot, yhat)], styles=['p', 'l'], labels=['yblind', 'yhat'])
  


if __name__ == '__main__':
  #log.basicConfig(filename='debug00.log', level=log.DEBUG)
  example_regress_on_time_stage_training_set.do_the_thing(True, 1500, 3)
  do_the_thing()


