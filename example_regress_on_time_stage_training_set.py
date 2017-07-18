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
FNAME_INORM = 'data_demo/inorm'
FNAME_ONORM = 'data_demo/onorm'
FNAME_TRAIN = 'data_demo/time_stage_lstm_train.npy'


def do_the_thing(fit=True, num_epochs=1500, num_neurons=3):  
    series = pd.read_csv('data_demo/shale_time_stage_exp.csv')
    #print('SERIES')
    #print(series.head())

    # we predict from previous y value, so features = labels with
    # a shift. first a 1D array
    y = series['y'].values
    #print('Y')
    #print(y, y.shape)

    stage =  series['stage'].values
    #print('STAGE')
    #print(stage, stage.shape)

    # stationary features (delta y), essentially start at time = 1, 
    # so one short of all labels
    ydeltaoutput = np.diff(y)
    #print('YDELTAOUTPUT')
    #print(ydeltaoutput, ydeltaoutput.shape)

    stagedeltainput = np.diff(stage)
    #print('STAGEDELTA')
    #print(stagedeltainput, stagedeltainput.shape)

    # bring the labels to shape
    yinput = y[:-1]
    #print('YINPUT')
    #print(yinput, yinput.shape)

    # normalize both
    input = np.transpose(np.array([yinput, stagedeltainput])) #yinput.reshape(-1, 1)
    normalizer_input = skprep.MinMaxScaler(feature_range=(-1, 1))
    input_normalized = normalizer_input.fit_transform(input)
    pickle.dump(normalizer_input, open(FNAME_INORM, "wb"))
    #print('INPUT NORM')
    #print(input_normalized, input_normalized.shape)
    ydeltaoutput = ydeltaoutput.reshape(-1, 1)
    normalizer_ydeltaoutput = skprep.MinMaxScaler(feature_range=(-1, 1))    
    ydelta_normalized = normalizer_ydeltaoutput.fit_transform(ydeltaoutput)
    pickle.dump(normalizer_ydeltaoutput, open(FNAME_ONORM, "wb"))    
    #print('YDELTAOUTPUT NORM')
    #print(ydelta_normalized, ydelta_normalized.shape)


    if fit:
        model = tbamd.lstm(input_normalized, ydelta_normalized, 1, num_epochs, num_neurons)
        tbamd.save(model, FNAME)
    else:
        model = tbamd.load(FNAME)

    yhat = [y[0]]
    for i in range(1, len(y)):
        # input is last value
        yprevious = yhat[-1]
        stagedeltacurrent = stagedeltainput[i-1]
        yinput = np.array([[yprevious, stagedeltacurrent]])
        yinput_normalized = normalizer_input.transform(yinput)
        yinput_normalized = yinput_normalized.reshape(yinput_normalized.shape[0], 1, 
        yinput_normalized.shape[1])
        ydelta_normalized = model.predict(yinput_normalized, batch_size=1)
        ydeltaoutput = normalizer_ydeltaoutput.inverse_transform(ydelta_normalized)
        ydeltaoutput = ydeltaoutput[0, 0]
        yhat += [yprevious+ydeltaoutput]

    
    xplot = series['x'].values
    np.save(FNAME_TRAIN, np.array([xplot, y]))
    tbapl.plot([(xplot, y), (xplot, yhat)], styles=['p', 'l'], labels=['ytrain', 'yhat'])

if __name__ == '__main__':
  do_the_thing()