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
    fname_csv = tbarc.rcparams['shale.lstm_stage.fnamecsv']
    name_dataset = fname_csv[10:-4]
    log.info(name_dataset)
    series = pd.read_csv(fname_csv)

    y = series['y'].values
    x = series['x'].values
    stage = series['stage'].values

    fname_model =  tbarc.rcparams['shale.lstm.sequence.fnamemodel'][:-3] + '_' + name_dataset + '.h5'
    fname_normalizer = tbarc.rcparams['shale.lstm.sequence.fnamenorm'][:-3] + '_' + name_dataset + '.h5'
    num_gradients_window = 3
    if 0:
        model, normalizerseq = tbamd.lstmseq(x, y, stage, 1000)
        tbamd.save(model, fname_model)
        pickle.dump(normalizerseq, open(fname_normalizer, 'wb'))
    else:
        model = tbamd.load(fname_model)
        normalizerseq = pickle.load(open(fname_normalizer, 'rb'))

    log.info("Inputs: {}".format(model.input_shape))
    log.info("Outputs: {}".format(model.output_shape))

    if 1:
        #stage = np.zeros_like(x)
        xpred, ypred = tbamd.predictseq(x, stage, normalizerseq, model)
        tbapl.plot([(x, y), (xpred, ypred)], styles=['p', 'l'], labels=['ytrain', 'yhat'])


if __name__ == '__main__':
    log.basicConfig(filename='debug00.log', level=log.DEBUG, filemode='w')
    do_the_thing()
