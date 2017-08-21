"""
Whole sequence lstm trained on multiple sequences, incl NA val
"""

import tinkerbell.app.plot as tbapl
import numpy as np
import sklearn.preprocessing as preproc
import pickle, sys, collections

import keras.models as kem
import keras.layers as kel


def l():
    sys.exit()


FNAME_MODEL = 'data_demo/model_post.h5'


def calc_exponential_decline(p0, exp, time):
    return p0 * np.exp(-exp * time)


def calc_two_stage_decline(p0, exp_stage_zero, exp_stage_one, time_max, time_next_stage, time_min=0., 
  production_jumpfactor=4., num=50, noise=0.1, noise_mean=1.):
    time = np.linspace(time_min, time_max, num)
    stage_one = np.where(time > time_next_stage)    
    production = calc_exponential_decline(p0, exp_stage_zero, time) * np.random.normal(noise_mean, noise, time.shape) 
    time_stage_one_relative = np.linspace(time_min, time_max-time_next_stage, len(time[stage_one]))
    production[stage_one] = calc_exponential_decline(production[stage_one][0] + p0/production_jumpfactor, exp_stage_one, 
      time_stage_one_relative) * np.random.normal(noise_mean, noise*2., time_stage_one_relative.shape)
    stage = np.zeros_like(time)
    stage[stage_one] = 1.
    production[production < 0.] = 0.
    return time, production, stage


def make_rnn(num_features, num_targets, num_timesteps, num_units):
    model = kem.Sequential()
    model.add(kel.LSTM(num_units, input_shape=(num_timesteps, num_features), return_sequences=True))
    model.add(kel.TimeDistributed(kel.Dense(num_targets, activation='sigmoid')))
    model.compile(loss='mse', optimizer='adam')
    return model


def plot_a_single_curve(y0_mean, d_stage_zero, d_stage_one, xmax):
    if 0:
        time, production, stage = calc_two_stage_decline(y0_mean, d_stage_zero, d_stage_one, xmax, 20., num=50)
        time1, production1, stage1 = calc_two_stage_decline(y0_mean, d_stage_zero, d_stage_one, xmax, 40., num=50)
        tbapl.plot([(time, production), (time1, production1)], ['p', 'p'])
    else:
        time, production, stage = calc_two_stage_decline(y0_mean, d_stage_zero, d_stage_one, xmax, 20., num=50)    
        tbapl.plot([(time, production)], ['p'],hide_labels=True, ylabel='production', xlabel='time', save_as='img\profile_example.svg',
          labels=['production'], secxyarraytuplesiterable=[(time, stage)], seclabels=['stage'], secstyles=['lstage'], secylim=(None, 3), secylabel='stage')
    


def do_the_thing(p0, exp_stage_zero, exp_stage_one, time_max):

    num_features = 2 # production rate
    num_timesteps = 50
    num_targets = 1
    num_units = 24

    num_discrete_stage_changes = 5
    num_realizations_per_stage_change = 10
    bounds_stage_change_time = (20., 40.)
    

    num_production_profiles = num_realizations_per_stage_change * num_discrete_stage_changes    
    # for each training sequence we provide it from first datapoint only to the next to last one
    num_sequences_per_profile = num_timesteps - 1
    num_sequences = num_production_profiles * num_sequences_per_profile

    ifeature_production = 0
    ifeature_stage = 1
    itarget_production = 0

    NA = -p0

    features = np.full((num_sequences, num_timesteps, num_features), NA)
    targets = np.full((num_sequences, num_timesteps, num_targets), NA)

    np.random.seed(42)
    isample = 0
    production = np.empty((num_timesteps, 1))
    stage = np.empty_like(production)
    for time_stage_change in np.linspace(*bounds_stage_change_time, num_discrete_stage_changes):
        for _ in range(num_realizations_per_stage_change):
            t, q, s = calc_two_stage_decline(p0, exp_stage_zero, exp_stage_one, time_max, time_stage_change, 
              num=num_timesteps) 
            for num_sample_points in range(1, num_timesteps):
                features[isample, :, ifeature_stage] = s[:] 
                features[isample, :num_sample_points, ifeature_production] = q[:num_sample_points]
                targets[isample, 1:, itarget_production] = q[1:] 
                isample += 1

    features = features.reshape(num_sequences*num_timesteps, num_features)
    normalizer_features = preproc.MinMaxScaler(feature_range=(0, 1), copy=False)
    normalizer_features.fit_transform(features)
    features = features.reshape(num_sequences, num_timesteps, num_features)

    targets = targets.reshape(num_sequences*num_timesteps, num_targets)
    normalizer_targets = preproc.MinMaxScaler(feature_range=(0, 1), copy=False)
    normalizer_targets.fit_transform(targets)
    targets = targets.reshape(num_sequences, num_timesteps, num_targets)
    
    if 0:
        print(features)
        print(targets)
        l()

    if 1:
        model = make_rnn(num_features, num_targets, num_timesteps, num_units)

        # batch_size should be whole denominator of num sequences per curve
        model.fit(features, targets, epochs=2, batch_size=5, validation_split=0.2)

        model.save(FNAME_MODEL)
    else:
        model = kem.load_model(FNAME_MODEL)


    time_stage_change = 20.
    time, production, stage = calc_two_stage_decline(p0, exp_stage_zero, exp_stage_one, time_max, time_stage_change, 
      num=num_timesteps)
    num_production_history = 6
    production_in = np.full((num_timesteps, 1), NA)
    production_in[:num_production_history, 0] = production[:num_production_history]
    stage_in = np.zeros_like(production_in)
    stage_in[:, 0] = stage[:]

    features = np.empty((1, num_timesteps, num_features))
    features[0, :, ifeature_production] = production_in[:, 0]
    features[0, :, ifeature_stage] = stage_in[:, 0]
    features = features.reshape(num_timesteps, num_features)
    normalizer_features.transform(features)
    features = features.reshape(1, num_timesteps, num_features)    
    targets = model.predict(features) 

    targets = targets.reshape(num_timesteps, num_targets)
    normalizer_targets.inverse_transform(targets)

    tbapl.plot([(time[1:], targets[1:, 0]), (time, production)], ['l', 'p'])
    sys.exit()


if __name__ == '__main__':
    p0 = 50.
    exp_stage_zero = 0.12
    exp_stage_one = 0.1
    time_max = 55.
    #plot_a_single_curve(p0, exp_stage_zero, exp_stage_one, time_max)
    do_the_thing(p0, exp_stage_zero, exp_stage_one, time_max)