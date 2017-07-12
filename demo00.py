#http://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import math
import sklearn.preprocessing as preproc

import keras.models as kem
import keras.layers as kel

VERBOSITY = 2

def dateparser(x):
    return pd.datetime.strptime('19'+x, '%Y-%b')

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
    return yhat + history.iat[-interval, 0]

def scale(train, test):
    # fit scaler
    scaler = preproc.MinMaxScaler(feature_range=(-1, 1))
    train = train.values.reshape(-1, 1)
    scaler = scaler.fit(train)
    # transform train    
    train_scaled = scaler.transform(train)
    # transform test
    test = test.values.reshape(-1, 1)
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
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    return model

series = pd.read_csv('shampoo-sales.csv', parse_dates=[0], index_col=0, date_parser=dateparser)
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


itrainingend = math.ceil(len(supervised)/3) 
train, test = supervised.iloc[:itrainingend, :-1], supervised.iloc[itrainingend:, -1:]
if VERBOSITY > 3:
    print('TRAIN-TEST')    
    print(itrainingend)    
    print(train)
    print(test)

scaler, train_scaled, test_scaled = scale(train, test)
if VERBOSITY > 3:
    print('TRAIN-TEST SCALED')  
    print(train_scaled)
    print(test_scaled)

fname_model = 'data_demo/model_lstm.h5'
if 1:
    model = fit_lstm(train_scaled, 1, 3000, 4)
    model.save(fname_model)
else:
    model = kem.load_model(fname_model)