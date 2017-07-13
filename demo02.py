import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import sklearn.preprocessing as skprep
import sklearn.metrics as skmet
from sys import exit
import keras.models as kem
import keras.layers as kel


def fit_lstm(features, labels, batch_size, nb_epoch, neurons):
    X, y = features, labels[:, 0]
    #print(y, y.shape)
    #exit()
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = kem.Sequential()
    model.add(kel.LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(kel.Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        print('EPOCH', i, '\\', nb_epoch)
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    return model


series = pd.read_csv('data_demo/shale_time_exp.csv')
print('SERIES')
print(series.head())

# we predict from previous y value, so features = labels wit
# a shift. first a 1D array
y = series.iloc[:, -1].values
print('Y')
print(y, y.shape)

# stationary features (delta y), essentially start at time = 1, 
# so one short of all labels
ydelta = np.diff(y)
print('YDELTA')
print(ydelta, ydelta.shape)

# bring the labels to shape
yinput = y[:-1]
print('YINPUT')
print(yinput, yinput.shape)

# normalize both
yinput = yinput.reshape(-1, 1)
normalizer_yinput = skprep.MinMaxScaler(feature_range=(-1, 1))
yinput_normalized = normalizer_yinput.fit_transform(yinput)
print('YINPUT NORM')
print(yinput_normalized, yinput_normalized.shape)
ydelta = ydelta.reshape(-1, 1)
normalizer_ydelta = skprep.MinMaxScaler(feature_range=(-1, 1))
ydelta_normalized = normalizer_ydelta.fit_transform(ydelta)
print('YDELTA NORM')
print(ydelta_normalized, ydelta_normalized.shape)


fname_model = 'data_demo/model_lstm_exp02.h5'
if 1:
    model = fit_lstm(yinput_normalized, ydelta_normalized, 1, 1000, 4)
    model.save(fname_model)
else:
    model = kem.load_model(fname_model)

yhat = [y[0]]
for i in range(1, len(y)-1):
    # input is last value
    yprevious = yhat[-1]
    yinput = np.array([[yprevious]])
    yinput_normalized = normalizer_yinput.transform(yinput)
    yinput_normalized = yinput_normalized.reshape(yinput_normalized.shape[0], 1, yinput_normalized.shape[1])
    ydelta_normalized = model.predict(yinput_normalized, batch_size=1)
    ydelta = normalizer_ydelta.inverse_transform(ydelta_normalized)
    ydelta = ydelta[0,0]
    yhat += [yprevious+ydelta]
    

plt.plot(yhat)
plt.plot(y)
plt.show()
