import numpy as np
import logging as log
import tinkerbell.app.make as tbamk
import tinkerbell.app.rcparams as tbarc
import tinkerbell.app.plot as tbapl
import tinkerbell.app.model as tbamd
import tinkerbell.domain.point as tbdpt


def do_the_thing():
  y0 = tbarc.rcparams['shale.lstm.y0_mean']*tbarc.rcparams['shale.lstm.y0_mult']
  d = tbarc.rcparams['shale.lstm_stage.d']
  xmax = tbarc.rcparams['shale.lstm_stage.xmax']
  num_points = tbarc.rcparams['shale.lstm_stage.num_points']
  xdisc = tbarc.rcparams['shale.lstm_stage.xdisc_mean'] + xmax*tbarc.rcparams['shale.lstm_stage.xdisc_mult']

  np.random.seed(42)
  pts, ixdisc = tbamk.points_exponential_discontinuous_declinelinear_noisy(y0, d, xmax, xdisc, num=num_points)

  xcomp_pts, ycomp_pts =  tbdpt.point_coordinates(pts)

  stage = np.zeros_like(ycomp_pts)
  stage[ixdisc:] = 1.0

  features = tbamd.Features(ycomp_pts, stage)
  targets = tbamd.Targets(ycomp_pts)

  fname_normalizer = tbarc.rcparams['shale.lstm_stage.fnamenormalizer']
  normalizer = tbamd.Normalizer.load(fname_normalizer)

  fname_model = tbarc.rcparams['shale.lstm_stage.fnamenmodel']
  model = tbamd.load(fname_model)

  yhat = tbamd.predict(y0, stage, normalizer, model)

  xplot = xcomp_pts[:-1]
  yref = ycomp_pts[:-1]
  fname_blinddata = tbarc.rcparams['shale.lstm_stage.fnamenblinddata']
  fname_preddata = tbarc.rcparams['shale.lstm_stage.fnamenpreddata']
  np.save(fname_blinddata, np.array([xplot, yref]))
  np.save(fname_preddata, np.array([xplot, yhat]))
  tbapl.plot([(xplot, yref), (xplot, yhat)], styles=['p', 'l'], labels=['yblind', 'yhat'])
  


if __name__ == '__main__':
  #log.basicConfig(filename='debug00.log', level=log.DEBUG)
  #example_regress_on_time_stage_training_set.do_the_thing(False, 1500, 3)
  do_the_thing()


