"""
Multilayer perceptron regression model
"""

import tinkerbell.app.plot as tbapl
import tinkerbell.domain.make as tbdmk
import tinkerbell.domain.point as tbdpt
import tinkerbell.app.make as tbamk
import tinkerbell.app.model as tbamd
import tinkerbell.app.rcparams as tbarc
import pandas as pd
import numpy as np
import sklearn.preprocessing as preproc
import pickle, sys, collections

from keras.models import Sequential, load_model
from keras.layers import Dense
import keras.callbacks as kec

FNAME_MODEL = 'data_demo/model_mplgradseq.h5'
FNAME_FNORM = 'data_demo/fnorm_mplgradseq'
FNAME_TNORM = 'data_demo/tnorm_mplgradseq'
XDISC_MIN = 20.0
XDISC_MAX = 40.0
XMAX = 70
NUM_PTS = 75
D = 0.1

def do_the_thing():

    num_features = NUM_PTS
    num_targets = NUM_PTS

    def model_deep():
        model = Sequential()
        afct = 'sigmoid'
        #afct = 'tanh'
        #afct = 'softmax'
        model.add(Dense(num_features, input_dim=num_features, activation=afct))
        model.add(Dense(num_features, activation=afct))
        model.add(Dense(num_targets, activation=afct))
        model.compile(loss='mse', optimizer='adam')
        return model

    num_xdisc = 5
    num_realizations = 5
    xdiscspace = (XDISC_MIN, XDISC_MAX)
    y0_mean = tbarc.rcparams['shale.exp.y0_mean']
    d = D
    k = tbarc.rcparams['shale.exp.k']
    xmax = XMAX

    num_sequences = num_realizations*num_xdisc
    num_samples = num_sequences
    features = np.zeros((num_samples, num_features))
    targets = np.zeros((num_samples, num_targets))

    np.random.seed(42)
    isample = 0
    for xdisc in np.linspace(*xdiscspace, num_xdisc):
        for irealization in range(num_realizations):
            y0 = y0_mean
            pts, ixdisc = tbamk.points_exponential_discontinuous_declinelinear_noisy(y0, d, xmax, xdisc, 
              num=NUM_PTS)
            stage = np.zeros((NUM_PTS,))
            stage[ixdisc:] = 1.0
            #stage_delta = np.diff(stage)
            #stage_delta_full = np.zeros((NUM_PTS,))
            #stage_delta_full[1:] = stage_delta[:]
            time, production = tbdpt.point_coordinates(pts)
            features[isample,:] = stage[:]
            targets[isample,:] = production[:]
            isample += 1

    print(features.shape, targets.shape)  

    if 0:
        normalizer_features = preproc.MinMaxScaler() 
        features_normalized = normalizer_features.fit_transform(features)
        pickle.dump(normalizer_features, open(FNAME_FNORM, "wb"))

        normalizer_targets = preproc.MinMaxScaler() 
        targets_normalized = normalizer_targets.fit_transform(targets)
        pickle.dump(normalizer_targets, open(FNAME_TNORM, "wb"))

        model = model_deep()

        num_epochs = 200

        History = collections.namedtuple("History", "history")
        
        with tbamd.ProgressBar(num_epochs) as progress_bar:
            progress_bar.iepoch = 0

            def advance_progress_bar(logs):
                history = History(logs)                
                progress_bar.update(progress_bar.iepoch, history)
                progress_bar.iepoch += 1
            
            after_epoch = kec.LambdaCallback(on_epoch_end=lambda batch, logs : advance_progress_bar(logs))

            model.fit(features_normalized, targets_normalized, epochs=num_epochs, batch_size=2,
              callbacks=[after_epoch], verbose=0)

        model.save(FNAME_MODEL)
    else:
        model = load_model(FNAME_MODEL)
        normalizer_features = pickle.load(open(FNAME_FNORM, "rb"))
        normalizer_targets = pickle.load(open(FNAME_TNORM, "rb"))

    #sys.exit()

    xdisc, att = np.mean(xdiscspace), '0'
    xdisc, att = np.max(xdiscspace)*1.2, '1'
    pts, ixdisc = tbamk.points_exponential_discontinuous_declinelinear_noisy(y0, d, xmax, xdisc, 
      num=NUM_PTS)
    time, production = tbdpt.point_coordinates(pts)
    stage = np.zeros_like(time)
    print(time[ixdisc])
    stage[ixdisc:] = 1
    stagers = stage.reshape((1,-1))
    stagers = normalizer_features.transform(stagers)
    production_hat = model.predict(stagers)
    production_hat = normalizer_targets.inverse_transform(production_hat)
    production_hat = production_hat[0]
    tbapl.plot([(time, production_hat), (time, production)], ['l', 'p'], 
      hide_labels=True, ylabel='production', xlabel='time', labels=['prediction', 'reference'],
      save_as='img/mlpseq'+att+'0.png', secylabel='stage')
    tbapl.plot([(time, production_hat), ], ['l'], 
      hide_labels=True, ylabel='production', xlabel='time', labels=['prediction'],
      save_as='img/mlpseq'+att+'1.png', secxyarraytuplesiterable=[(time, stage)], seclabels=['planned stage'],
      secstyles=['lstage'], secylim=(None, 3), secylabel='stage')





if __name__ == '__main__':
    do_the_thing()