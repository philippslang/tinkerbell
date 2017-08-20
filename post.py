"""
Whole sequence lstm trained on multiple sequences, incl NA val
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

import keras.models as kem
import keras.layers as kel
import keras.callbacks as kec


def l():
    sys.exit()


FNAME_MODEL = 'data_demo/model_post.h5'
FNAME_NORM = 'data_demo/norm_post'
XDISC_MIN = 20.0
XDISC_MAX = 40.0
XMAX = 70.0
NUM_PTS = 50 + 1
D = 0.1

def do_the_thing():

    num_features = 2
    num_timesteps = NUM_PTS - 1
    num_targets = 1
    num_units = 25

    def make_model():
        model = kem.Sequential()
        model.add(kel.LSTM(num_units, input_shape=(num_timesteps, num_features), return_sequences=True))
        model.add(kel.TimeDistributed(kel.Dense(num_targets)))
        model.compile(loss='mse', optimizer='adam')
        return model

    num_xdisc = 4
    num_realizations_per_xdisc = 8
    xdiscspace = (XDISC_MIN, XDISC_MAX)
    y0_mean = tbarc.rcparams['shale.exp.y0_mean']
    d = D
    k = tbarc.rcparams['shale.exp.k']
    xmax = XMAX

    num_production_profiles = num_realizations_per_xdisc * num_xdisc    
    # for each training sequence we provide it from first datapoint only to the next to last one
    num_sequences_per_profile = num_timesteps - 1
    num_sequences = num_production_profiles * num_sequences_per_profile
    print(num_sequences_per_profile)

    ifeature_production = 0
    ifeature_stage = 1
    itarget_production = 0
    
    normalizer_production = preproc.MinMaxScaler(feature_range=(0, 1), copy=False)
    range_production = np.array([[xmax], [-xmax]])
    normalizer_production.fit_transform(range_production)
    normalizer_stage = preproc.MinMaxScaler(feature_range=(0, 1), copy=False) 
    range_stage = [[1.0], [0.0]]
    normalizer_stage.fit(range_stage)

    NA = range_production.min()

    x = np.full((num_sequences, num_timesteps, num_features), NA)
    y = np.full((num_sequences, num_timesteps, num_targets), NA)

    np.random.seed(42)
    isample = 0
    production = np.empty((NUM_PTS, 1))
    stage = np.empty_like(production)
    for xdisc in np.linspace(*xdiscspace, num_xdisc):
        for _ in range(num_realizations_per_xdisc):
            y0 = y0_mean
            pts, ixdisc = tbamk.points_exponential_discontinuous_declinelinear_noisy(y0, d, xmax, xdisc, 
              num=NUM_PTS)
            _, q = tbdpt.point_coordinates(pts)  
            production[:, 0] = q[:]
            stage[:ixdisc, 0] = 0.0
            stage[ixdisc:, 0] = 1.0
            normalizer_production.transform(production)
            normalizer_stage.transform(stage)
            for num_sample_points in range(1, num_timesteps):
                x[isample, :, ifeature_stage] = stage[1:, 0]
                x[isample, :num_sample_points, ifeature_production] = production[:num_sample_points, 0]
                y[isample, :, itarget_production] = production[1:, 0]
                isample += 1

    if 0:
        print(x)
        print(y)
        l()

    if 1:
        model = make_model()

        # batch_size should be whole denominator of num sequences per curve
        model.fit(x, y, epochs=2, batch_size=2)

        model.save(FNAME_MODEL)
    else:
        model = kem.load_model(FNAME_MODEL)


    xdisc = xdiscspace[0] + np.diff(xdiscspace)[0] * 0.2
    pts, ixdisc = tbamk.points_exponential_discontinuous_declinelinear_noisy(y0, d, xmax, xdisc, 
      num=NUM_PTS)
    time, production = tbdpt.point_coordinates(pts)
    num_production_history = 6
    production_in = np.full((num_timesteps, 1), NA)
    production_in[:num_production_history, 0] = production[:num_production_history]
    stage_in= np.zeros_like(production_in)
    stage_in[ixdisc:, 0] = 1.0
    
    normalizer_stage.transform(stage_in)
    normalizer_production.transform(production_in)
    xin = np.empty((1, num_timesteps, num_features))
    xin[0, :, ifeature_production] = production_in[:, 0]
    xin[0, :, ifeature_stage] = stage_in[:, 0]
    y_hat = model.predict(xin) 
    normalizer_production.inverse_transform(y_hat[0])
    tbapl.plot([(time[1:-1], y_hat[0, 1:, 0]), (time[:num_production_history], production[:num_production_history])], ['l', 'p'])
    tbapl.plot([(time[1:-1], y_hat[0, 1:, 0]), (time, production)], ['l', 'p'])
    sys.exit()


    tbapl.plot([(time, production_hat), (time, production)], ['l', 'p'], 
      hide_labels=True, ylabel='production', xlabel='time', labels=['prediction', 'reference'],
      save_as='img/mlpseq'+att+'0.png', secylabel='stage')

    sys.exit()
    tbapl.plot([(time, production_hat), ], ['l'], 
      hide_labels=True, ylabel='production', xlabel='time', labels=['prediction'],
      save_as='img/mlpseq'+att+'1.png', secxyarraytuplesiterable=[(time, stage)], seclabels=['planned stage'],
      secstyles=['lstage'], secylim=(None, 3), secylabel='stage')





if __name__ == '__main__':
    do_the_thing()