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

FNAME_MODEL = 'data_demo/model_mplgrad.h5'
FNAME_FNORM = 'data_demo/fnorm_mplgrad'
FNAME_TNORM = 'data_demo/tnorm_mplgrad'
XDISC_MIN = 20.0
XDISC_MAX = 40.0
XMAX = 70
NUM_PTS = 75
D = 0.1

def do_the_thing():

    size_window = 3
    num_features = size_window*2
    num_targets = 1

    def model_deep():
        model = Sequential()
        afct = 'sigmoid'
        #afct = 'tanh'
        #afct = 'softmax'
        model.add(Dense(num_features, input_dim=num_features, activation=afct))
        model.add(Dense(size_window, activation=afct))
        #model.add(Dense(num_features, activation=afct))
        #model.add(Dense(size_window, activation=afct))
        #model.add(Dense(num_targets, activation=afct))
        model.add(Dense(num_targets, activation=afct))
        model.compile(loss='mse', optimizer='adam')
        return model

    num_xdisc = 5
    num_realizations = 3
    xdiscspace = (XDISC_MIN, XDISC_MAX)
    y0_mean = tbarc.rcparams['shale.exp.y0_mean']
    d = D
    k = tbarc.rcparams['shale.exp.k']
    xmax = XMAX

    plot_data = False # True
    allpts = []
    allstages = []
    plot_pts = []

    np.random.seed(42)
    for xdisc in np.linspace(*xdiscspace, num_xdisc):
        for irealization in range(num_realizations):
            y0 = y0_mean
            pts, ixdisc = tbamk.points_exponential_discontinuous_declinelinear_noisy(y0, d, xmax, xdisc, 
              num=NUM_PTS)
            allpts += [pts]
            stages = np.zeros((NUM_PTS,))
            stages[ixdisc:] = 1.0
            allstages += [stages]
            if plot_data:
                plot_pts += [tbdpt.point_coordinates(pts)]

    if plot_data:
        tbapl.plot(plot_pts, styles=['l']*len(plot_pts), ylabel='production', xlabel='time', hide_labels=True,
          save_as='img/mplgrad_training.png')
        sys.exit()

    assert len(allpts) == len(allstages)

    offset_pred = 1
    num_sequences = len(allpts)
    features = []
    targets = []

    for isequence in range(num_sequences):
        time, production = tbdpt.point_coordinates(allpts[isequence])
        dp_dt = np.diff(production) / np.diff(time)
        stage_delta = np.diff(allstages[isequence])
        for idiff in range(len(dp_dt)-offset_pred-size_window):
            features_diff = []
            targets_diff = []
            for iwindow in range(size_window):
                features_diff += [dp_dt[idiff+iwindow], stage_delta[idiff+iwindow]]
            targets_diff += [dp_dt[idiff+size_window]]
            features += [features_diff]
            targets += [targets_diff]

    features = np.array(features)
    targets = np.array(targets)

    print(features.shape, targets.shape)  

    np.savetxt('mlpft.txt', features)  
    np.savetxt('mlptr.txt', targets)  

    if 1:
        normalizer_features = preproc.MinMaxScaler() 
        features_normalized = normalizer_features.fit_transform(features)
        pickle.dump(normalizer_features, open(FNAME_FNORM, "wb"))

        normalizer_targets = preproc.MinMaxScaler() 
        targets_normalized = normalizer_targets.fit_transform(targets)
        pickle.dump(normalizer_targets, open(FNAME_TNORM, "wb"))

        model = model_deep()

        num_epochs = 500

        History = collections.namedtuple("History", "history")
        
        with tbamd.ProgressBar(num_epochs) as progress_bar:
            progress_bar.iepoch = 0

            def advance_progress_bar(logs):
                history = History(logs)                
                progress_bar.update(progress_bar.iepoch, history)
                progress_bar.iepoch += 1
            
            after_epoch = kec.LambdaCallback(on_epoch_end=lambda batch, logs : advance_progress_bar(logs))

            model.fit(features_normalized, targets_normalized, epochs=num_epochs, batch_size=1,
              callbacks=[after_epoch], verbose=0)

        model.save(FNAME_MODEL)
    else:
        model = load_model(FNAME_MODEL)
        normalizer_features = pickle.load(open(FNAME_FNORM, "rb"))
        normalizer_targets = pickle.load(open(FNAME_TNORM, "rb"))

    #sys.exit()

    xdisc = np.mean(xdiscspace)
    pts, ixdisc = tbamk.points_exponential_discontinuous_declinelinear_noisy(y0, d, xmax, xdisc, 
      num=NUM_PTS)
    time, production = tbdpt.point_coordinates(pts)
    production_hat = list(production[:size_window+1]) # account for diff
    stage = np.zeros_like(time)
    stage[ixdisc:] = 1
    for istep in range(size_window+1, len(time)-size_window-offset_pred):
        production_window = production_hat[istep-size_window-1:istep]
        time_window = time[istep-size_window-1:istep]
        stage_window = stage[istep-size_window-1:istep]
        dp_dt_window = np.diff(production_window) / np.diff(time_window)
        stage_delta_window = np.diff(stage_window)
        features_window = []
        for iwindow in range(len(dp_dt_window)):
            features_window += [dp_dt_window[iwindow], stage_delta_window[iwindow]]
        features_window = np.array(features_window)
        features_window = features_window.reshape((1,-1))
        features_window = normalizer_features.transform(features_window)
        target_window = model.predict(features_window)
        target_window = normalizer_targets.inverse_transform(target_window)
        dp_dt_step = target_window[0, -1]
        time_delta = time[istep] - time[istep-1]
        production_hat += [production_hat[-1]+time_delta*dp_dt_step]
        #print(production_hat)
    
    tbapl.plot([(time[:len(production_hat)], production_hat), (time, production)], ['p', 'p'], 
      hide_labels=True, ylabel='production', xlabel='time')





if __name__ == '__main__':
    do_the_thing()