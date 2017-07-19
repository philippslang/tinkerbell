"""
Recurrent neural network model with stages.

TODO
train on multiple curves
experiment with lag (diff n)
experiment with number of features (differences of 2, 4, 6 last observations; 
  these would also have to include the stage value)
ditto number neurons (> 3)
ditto number of layers, I'd expect one needed per disc
moving average gradient
optimizing for the error metric (mean absolute error) instead of RMSE
change timesteps 
update model 
"""

import pandas as pd
import numpy as np
import logging as log
import tinkerbell.app.plot as tbapl
import tinkerbell.app.model as tbamd
import tinkerbell.app.rcparams as tbarc

#FNAME = 'data_demo/model_lstm_stages_exp.h5'
#FNAME_TRAIN = 'data_demo/time_stage_lstm_train.npy'


def do_the_thing(fit=True, num_epochs=1500, num_neurons=3):  
    series = pd.read_csv(tbarc.rcparams['shale.lstm_stage.fnamecsv'])
    #print(series.head())

    y = series['y'].values
    stage = series['stage'].values
    features = tbamd.Features(y, stage)
    targets = tbamd.Targets(y)

    # normalize both
    normalizer = tbamd.Normalizer.fit(features, targets) 
    fname_normalizer = tbarc.rcparams['shale.lstm_stage.fnamenormalizer']
    normalizer.save(fname_normalizer)
    input_normalized = normalizer.normalize_features(features)  
    ydelta_normalized = normalizer.normalize_targets(targets)

    fname_model = tbarc.rcparams['shale.lstm_stage.fnamenmodel']
    if fit:
        model = tbamd.lstm(input_normalized, ydelta_normalized, 1, num_epochs, num_neurons)
        tbamd.save(model, fname_model)
    else:
        model = tbamd.load(fname_model)

    yhat = tbamd.predict(y[0], stage, normalizer, model)
    
    xplot = series['x'].values[:-1]
    yref = y[:-1]
    fname_traindata = tbarc.rcparams['shale.lstm_stage.fnamentraindata']
    np.save(fname_traindata, np.array([xplot, yref]))
    tbapl.plot([(xplot, yref), (xplot, yhat)], styles=['p', 'l'], labels=['ytrain', 'yhat'])


if __name__ == '__main__':
  do_the_thing(False, 100, 3)