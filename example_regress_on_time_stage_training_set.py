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
    # stage is misleading it's stage delta
    stage = series['stage'].values
    features = tbamd.Features(y, stage)
    targets = tbamd.Targets(y)

    # normalize both
    normalizer = tbamd.Normalizer.fit(features, targets) 
    normalizer.save(FNAME_NORM)
    input_normalized = normalizer.normalize_features(features)  
    ydelta_normalized = normalizer.normalize_targets(targets)

    if fit:
        model = tbamd.lstm(input_normalized, ydelta_normalized, 1, num_epochs, num_neurons)
        tbamd.save(model, FNAME)
    else:
        model = tbamd.load(FNAME)

    yhat = [y[0]]
    for i in range(1, len(y)-1):
        # input is last value
        ylasttwo = y[i-1:i+1]
        stagelasttwo = stage[i-1:i+1]
        features_predict = tbamd.Features(ylasttwo, stagelasttwo)
        features_predict_normalized = normalizer.normalize_features(features_predict)
        features_predict_normalized = features_predict_normalized.reshape(features_predict_normalized.shape[0], 
          1, features_predict_normalized.shape[1])
        target_predicted_normalized = model.predict(features_predict_normalized, batch_size=1)
        target_predicted = normalizer.denormalize_targets(target_predicted_normalized)
        target_predicted = target_predicted[0, 0]
        yhat += [yhat[-1]+target_predicted]

    
    xplot = series['x'].values[:-1]
    yref = y[:-1]
    np.save(FNAME_TRAIN, np.array([xplot, y]))
    tbapl.plot([(xplot, yref), (xplot, yhat)], styles=['p', 'l'], labels=['ytrain', 'yhat'])

if __name__ == '__main__':
  do_the_thing(True, 1500, 3)