"""
Recurrent neural network model with stages

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
import sklearn.preprocessing as skprep
import sklearn.metrics as skmet
from sys import exit
import logging as log
import tinkerbell.app.plot as tbapl
import tinkerbell.app.model as tbamd
import pickle

FNAME = 'data_demo/model_lstm_stages_exp.h5'
FNAME_NORM = 'data_demo/norm'
FNAME_TRAIN = 'data_demo/time_stage_lstm_train.npy'


def do_the_thing(fit=True, num_epochs=1500, num_neurons=3):  
    series = pd.read_csv('data_demo/shale_time_stage_exp.csv')
    #print(series.head())

    y = series['y'].values
    stage = series['stage'].values
    features = tbamd.Features(y, stage)
    targets = tbamd.Targets(y)

    # bring the labels to shape
    #yinput = features.production[:-1]

    # normalize both
    input = features.matrix() 
    ydeltaoutput = targets.matrix()
    normalizer = tbamd.Normalizer.fit(input, ydeltaoutput) 
    normalizer.save(FNAME_NORM)
    input_normalized = normalizer.normalize_features(input)  
    ydelta_normalized = normalizer.normalize_targets(ydeltaoutput)

    if fit:
        model = tbamd.lstm(input_normalized, ydelta_normalized, 1, num_epochs, num_neurons)
        tbamd.save(model, FNAME)
    else:
        model = tbamd.load(FNAME)

    yhat = [y[0]]
    for i in range(1, len(y)):
        # input is last value
        yprevious = yhat[-1]
        stagedeltacurrent = features.dstage_dstep[i-1]
        yinput = np.array([[yprevious, stagedeltacurrent]])
        yinput_normalized = normalizer.normalize_features(yinput)
        yinput_normalized = yinput_normalized.reshape(yinput_normalized.shape[0], 1, 
        yinput_normalized.shape[1])
        ydelta_normalized = model.predict(yinput_normalized, batch_size=1)
        ydeltaoutput = normalizer.denormalize_targets(ydelta_normalized)
        ydeltaoutput = ydeltaoutput[0, 0]
        yhat += [yprevious+ydeltaoutput]

    
    xplot = series['x'].values
    np.save(FNAME_TRAIN, np.array([xplot, y]))
    tbapl.plot([(xplot, y), (xplot, yhat)], styles=['p', 'l'], labels=['ytrain', 'yhat'])

if __name__ == '__main__':
  do_the_thing(True, 99, 3)