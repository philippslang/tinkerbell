"""
Recurrent neural network model example with one-stage production data.
"""
import sys
import pandas as pd
import numpy as np
import logging as log
import pickle
import tinkerbell.app.plot as tbapl
import tinkerbell.app.model as tbamd
import tinkerbell.app.rcparams as tbarc


def do_the_thing():
    fname_csv = tbarc.rcparams['shale.lstm.fnamecsv']
    #fname_csv = tbarc.rcparams['shale.lstm_stage.fnamecsv']
    name_dataset = fname_csv[10:-4]
    log.info(name_dataset)
    series = pd.read_csv(fname_csv)

    y = series['y'].values
    x = series['x'].values
    stage = series['stage'].values
 
    fname_model =  tbarc.rcparams['shale.lstm.sequence.win.fnamemodel'][:-3] + '_' + name_dataset + '.h5'
    fname_normalizer = tbarc.rcparams['shale.lstm.sequence.win.fnamenorm'][:-3] + '_' + name_dataset + '.h5'
    num_timesteps = 3
    num_units = 3
    num_epochs = 500
    if 1:
        model, normalizer = tbamd.lstmseqwin(y, stage, num_epochs, num_timesteps, num_units)
        tbamd.save(model, fname_model)
        pickle.dump(normalizer, open(fname_normalizer, 'wb'))
    else:
        model = tbamd.load(fname_model)
        normalizer = pickle.load(open(fname_normalizer, 'rb'))

    log.info("Inputs: {}".format(model.input_shape))
    log.info("Outputs: {}".format(model.output_shape))

    if 1:
        ypred = tbamd.predictseqwin(y[:num_timesteps], stage, normalizer, model)
        xpred = x[:len(ypred)]
        tbapl.plot([(x, y), (xpred, ypred)], styles=['p', 'l'], labels=['ytrain', 'yhat'])


if __name__ == '__main__':
    log.basicConfig(filename='debug01.log', level=log.DEBUG, filemode='w')
    do_the_thing()
