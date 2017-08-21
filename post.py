"""
Whole sequence lstm trained on multiple sequences, incl NA val
"""

import tinkerbell.app.plot as tbapl
import numpy as np
import sklearn.preprocessing as preproc
import pickle, sys, collections

import keras.models as kem
import keras.layers as kel
import keras.callbacks as kec


def l():
    sys.exit()


FNAME_MODEL = 'data_demo/model_post.h5'


def calc_exponential_decline(p0, exp, time):
    return p0 * np.exp(-exp * time)


def calc_two_stage_decline(p0, exp_stage_zero, exp_stage_one, time_max, time_next_stage, time_min=0., 
  production_jumpfactor=30.0, num=50, noise=0.1, noise_mean=1.):
    time = np.linspace(time_min, time_max, num)
    next_stage = np.where(time > time_next_stage)    
    production = calc_exponential_decline(p0, exp_stage_zero, time) * np.random.normal(noise_mean, noise, time.shape) 
    time_stage_one_relative = np.linspace(time_min, time_max-time_next_stage, len(time[next_stage]))
    production[next_stage] = calc_exponential_decline(np.sqrt(production[next_stage][0]**2 * production_jumpfactor) * 
      np.random.normal(noise_mean, noise*3), exp_stage_one, time_stage_one_relative) * np.random.normal(noise_mean, noise*2, time_stage_one_relative.shape)
    stage = np.zeros_like(time)
    stage[next_stage] = 1.0
    production[production < 0.] = 0.
    return time, production, stage


def make_rnn(num_features, num_targets, num_timesteps, num_units):
    model = kem.Sequential()
    model.add(kel.LSTM(num_units, input_shape=(num_timesteps, num_features), return_sequences=True))
    model.add(kel.TimeDistributed(kel.Dense(num_targets, activation='sigmoid')))
    model.compile(loss='mse', optimizer='adam')
    return model


def plot_a_single_curve():
    exp0 = 0.15
    exp1 = 0.1
    time, production, stage = calc_two_stage_decline(50., exp0, exp1, 55., 20., num=50)
    time1, production1, stage1 = calc_two_stage_decline(50., exp0, exp1, 55., 40., num=50)
    tbapl.plot([(time, production), (time1, production1)], ['p', 'p'])
    


def do_the_thing():

    num_features = 2
    num_timesteps = 50
    num_targets = 1
    num_units = 24

    num_stage_change_time = 5
    num_realizations_per_stage_change_time = 10
    xdiscspace = (20., 40.)
    y0_mean = 50.
    d_stage_zero = 0.15
    d_stage_one = 0.1
    xmax = 55.

    num_production_profiles = num_realizations_per_stage_change_time * num_stage_change_time    
    # for each training sequence we provide it from first datapoint only to the next to last one
    num_sequences_per_profile = num_timesteps - 1
    num_sequences = num_production_profiles * num_sequences_per_profile

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
    production = np.empty((num_timesteps, 1))
    stage = np.empty_like(production)
    for xdisc in np.linspace(*xdiscspace, num_stage_change_time):
        for _ in range(num_realizations_per_stage_change_time):
            y0 = y0_mean
            t, q, s = calc_two_stage_decline(y0, d_stage_zero, d_stage_one, xmax, xdisc, 
              num=num_timesteps) 
            production[:, 0] = q[:]
            stage[:, 0] = s[:]
            normalizer_production.transform(production)
            normalizer_stage.transform(stage)
            for num_sample_points in range(1, num_timesteps):
                x[isample, :, ifeature_stage] = stage[:, 0]
                x[isample, :num_sample_points, ifeature_production] = production[:num_sample_points, 0]
                y[isample, :, itarget_production] = production[:, 0]
                isample += 1

    if 0:
        print(x)
        print(y)
        l()

    if 0:
        model = make_rnn(num_features, num_targets, num_timesteps, num_units)

        # batch_size should be whole denominator of num sequences per curve
        model.fit(x, y, epochs=2, batch_size=5, validation_split=0.2)

        model.save(FNAME_MODEL)
    else:
        model = kem.load_model(FNAME_MODEL)


    xdisc = 20.0
    time, production, stage = calc_two_stage_decline(y0, d_stage_zero, d_stage_one, xmax, xdisc, 
      num=num_timesteps)
    num_production_history = 6
    production_in = np.full((num_timesteps, 1), NA)
    production_in[:num_production_history, 0] = production[:num_production_history]
    stage_in = np.zeros_like(production_in)
    stage_in[:, 0] = stage[:]

    normalizer_stage.transform(stage_in)
    normalizer_production.transform(production_in)
    xin = np.empty((1, num_timesteps, num_features))
    xin[0, :, ifeature_production] = production_in[:, 0]
    xin[0, :, ifeature_stage] = stage_in[:, 0]
    y_hat = model.predict(xin) 

    normalizer_production.inverse_transform(y_hat[0])

    #tbapl.plot([(time[1:], y_hat[0, 1:, 0]), (time[:num_production_history], production[:num_production_history])], ['l', 'p'])
    tbapl.plot([(time[1:], y_hat[0, 1:, 0]), (time, production)], ['l', 'p'])
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
    #plot_a_single_curve()
    do_the_thing()