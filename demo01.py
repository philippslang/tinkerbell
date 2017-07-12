#http://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import math
import sklearn.preprocessing as skprep
import sklearn.metrics as skmet
from sys import exit
import keras.models as kem
import keras.layers as kel

VERBOSITY = 1


def dateparser(x):
    return pd.datetime.strptime(x, '%y')

def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset.iat[i, 0] - dataset.iat[i - interval, 0]
        diff.append(value)
    return pd.Series(diff)

def inverse_difference(history, yhat, interval=1):
    try:
        val = yhat + history.iat[-interval, 0]
    except:
        val = yhat + history[-interval, 0]
    return val

def scale(train, test):
    # fit scaler
    scaler = skprep.MinMaxScaler(feature_range=(-1, 1))
    train = train.values.reshape(train.shape[0], train.shape[1])
    scaler = scaler.fit(train)
    # transform train    
    train_scaled = scaler.transform(train)
    # transform test
    test = test.values.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled
 
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = kem.Sequential()
    model.add(kel.LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(kel.Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        verbose = VERBOSITY > 2
        if VERBOSITY > 0:
            print('EPOCH', i, '\\', nb_epoch)
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=verbose, shuffle=False)
        model.reset_states()
    return model

def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0]

def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    vals = np.array(new_row)
    vals = vals.reshape(1, len(vals))
    inverted = scaler.inverse_transform(vals)
    return inverted[0, -1]


series = pd.read_csv('time_demo.csv', parse_dates=[0], index_col=0, date_parser=dateparser)
itrainingend = 2#math.ceil(len(series)/4)
raw_values = series.values
if VERBOSITY > 1:
    print('RAW')
    print(series.head())
if VERBOSITY > 3:
    series.plot()
    plt.show()

# excludes first
stationary = difference(series, 1)
if VERBOSITY > 1:
    print('STATIONARY')    
    print(stationary.head())
if VERBOSITY > 3:
    stationary.plot()
    plt.show()

# excludes first
inverted = []
for i in range(len(stationary)):
    value = inverse_difference(series, stationary.iat[i], len(series)-i)
    inverted.append(value)
inverted = pd.Series(inverted)
if VERBOSITY > 1:
    print('INVERTED')   
    print(inverted.head())
if VERBOSITY > 3:
    inverted.plot()
    plt.show()

supervised = timeseries_to_supervised(stationary, 1)

if VERBOSITY > 1:
    print('SUPERVISED')        
    print(supervised.head())
 

train, test = supervised.iloc[:itrainingend-1, :], supervised.iloc[itrainingend-1:, :] # since this is one short (first)
if VERBOSITY > 3:
    print('TRAIN-TEST')    
    print(itrainingend)    
    print(train)
    print(train.shape)
    print(test)

scaler, train_scaled, test_scaled = scale(train, test)
if VERBOSITY > 3:
    print('TRAIN-TEST SCALED')  
    print(train_scaled)
    print(test_scaled)


fname_model = 'data_demo/model_lstm_exp.h5'
if 1:
    model = fit_lstm(train_scaled, 1, 1500, 3)
    model.save(fname_model)
else:
    model = kem.load_model(fname_model)

#train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
#model.predict(train_reshaped, batch_size=1)

predictions = []
for i in range(len(test_scaled)):
    # make one-step forecast
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = forecast_lstm(model, 1, X)
    #yhat = y
    # invert scaling
    yhat = invert_scale(scaler, X, yhat)
    # invert differencing
    yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
    # store forecast
    predictions.append(yhat)
    expected = raw_values[len(train) + i + 1]
    print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))
 
# report performance
rmse = math.sqrt(skmet.mean_squared_error(raw_values[itrainingend:], predictions))
print('Test RMSE: %.3f' % rmse)
# line plot of observed vs predicted
plt.plot(raw_values[itrainingend:])
plt.plot(predictions)
plt.show()