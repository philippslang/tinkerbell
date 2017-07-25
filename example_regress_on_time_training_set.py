"""
Recurrent neural network model example with one-stage production data.
"""

import pandas as pd
import numpy as np
import tinkerbell.app.plot as tbapl
import tinkerbell.app.model as tbamd
import tinkerbell.app.rcparams as tbarc


def do_the_thing():
    series = pd.read_csv(tbarc.rcparams['shale.lstm.fnamecsv'])

    y = series['y'].values
    stage = np.zeros_like(y)
    features = tbamd.Features(y, stage)
    targets = tbamd.Targets(y)

    # normalize both
    normalizer = tbamd.Normalizer.fit(features, targets) 
    input_normalized = normalizer.normalize_features(features)  
    ydelta_normalized = normalizer.normalize_targets(targets)

    fname_model = tbarc.rcparams['shale.lstm.fnamemodel']
    if 1:
        model = tbamd.lstm(input_normalized, ydelta_normalized, 1, 250, 3)
        tbamd.save(model, fname_model)
    else:
        model = tbamd.load(fname_model)

    yhat = tbamd.predict(y[0], stage, normalizer, model)

    xplot = series['x'].values[:-1]
    yplot = y[:-1]
    tbapl.plot([(xplot, yplot), (xplot, yhat)], styles=['p', 'l'], labels=['ytrain', 'yhat'])


if __name__ == '__main__':
    do_the_thing()
