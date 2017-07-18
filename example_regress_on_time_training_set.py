"""
Recurrent neural network model
"""

import pandas as pd
import numpy as np
import sklearn.preprocessing as skprep
import sklearn.metrics as skmet
from sys import exit
import keras.models as kem
import keras.layers as kel
import tinkerbell.app.plot as tbapl
import tinkerbell.app.model as tbamd



series = pd.read_csv('data_demo/shale_time_exp.csv')
#print('SERIES')
#print(series.head())

# we predict from previous y value, so features = labels with
# a shift. first a 1D array
y = series.iloc[:, -1].values
#print('Y')
#print(y, y.shape)

# stationary features (delta y), essentially start at time = 1, 
# so one short of all labels
ydelta = np.diff(y)
#print('YDELTA')
#print(ydelta, ydelta.shape)

# bring the labels to shape
yinput = y[:-1]
#print('YINPUT')
#print(yinput, yinput.shape)

# normalize both
yinput = yinput.reshape(-1, 1)
normalizer_yinput = skprep.MinMaxScaler(feature_range=(-1, 1))
yinput_normalized = normalizer_yinput.fit_transform(yinput)
#print('YINPUT NORM')
#print(yinput_normalized, yinput_normalized.shape)
ydelta = ydelta.reshape(-1, 1)
normalizer_ydelta = skprep.MinMaxScaler(feature_range=(-1, 1))
ydelta_normalized = normalizer_ydelta.fit_transform(ydelta)
#print('YDELTA NORM')
#print(ydelta_normalized, ydelta_normalized.shape)


fname_model = 'data_demo/model_lstm_exp.h5'
if 1:
    model = tbamd.lstm(yinput_normalized, ydelta_normalized, 1, 500, 3)
    tbamd.save(model, fname_model)
else:
    model = tbamd.load(fname_model)

yhat = [y[0]]
for i in range(1, len(y)):
    # input is last value
    yprevious = yhat[-1]
    yinput = np.array([[yprevious]])
    yinput_normalized = normalizer_yinput.transform(yinput)
    yinput_normalized = yinput_normalized.reshape(yinput_normalized.shape[0], 1, 
      yinput_normalized.shape[1])
    ydelta_normalized = model.predict(yinput_normalized, batch_size=1)
    ydelta = normalizer_ydelta.inverse_transform(ydelta_normalized)
    ydelta = ydelta[0, 0]
    yhat += [yprevious+ydelta]

xplot = series['x'].values
tbapl.plot([(xplot, y), (xplot, yhat)], styles=['p', 'l'], labels=['y', 'yhat'])
