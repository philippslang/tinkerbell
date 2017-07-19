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

  yhat = tbamd.predict(y0, stage, normalizer, model)

  xplot = xcomp_pts[:-1]
  yref = ycomp_pts[:-1]
  np.save(FNAME_BLIND, np.array([xplot, yref]))
  np.save(FNAME_PRED, np.array([xplot, yhat]))
  tbapl.plot([(xplot, yref), (xplot, yhat)], styles=['p', 'l'], labels=['yblind', 'yhat'])
  


if __name__ == '__main__':
  #log.basicConfig(filename='debug00.log', level=log.DEBUG)
  example_regress_on_time_stage_training_set.do_the_thing(False, 1500, 3)
  do_the_thing()


