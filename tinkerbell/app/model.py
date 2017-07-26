import numpy as np
import pickle
import sys
import logging as log
import keras.models as kem
import keras.layers as kel
import sklearn.preprocessing as skprep
import collections as coll

def makes_deep_copy(fct):
    def ret_fct(*args, **kwargs):
        print('Think about this deep copy of a potentially large buffer in \'{}()\''.format(fct.__name__))
        return fct(*args, **kwargs)
    ret_fct.f = fct.__name__
    ret_fct.__doc__ = fct.__doc__
    ret_fct.__dict__.update(fct.__dict__)
    return ret_fct


class Features:
    def __init__(self, production, stage):        
        assert len(production) == len(stage), "Feature vectors must have same number of samples."
        self.production = np.copy(production)
        self.stage = np.copy(stage)
        self.eval_gradients()

    def eval_gradients(self):
        self.stage_delta = np.diff(self.stage)

    def matrix(self):
        # here we must account for that we lost the bottom row
        # when taking the delta from the production stage
        # this is coupled with matrix() in Targets in a sense
        # through the diff in the targets (we predict gradients)
        return np.transpose(np.array([self.production[:-1], self.stage_delta]))


class Targets:
    def __init__(self, production, time=None):
        """
        If time=None, assumes equidistant.
        """
        self.production = np.copy(production)        
        if time is not None:
            self.time = np.copy(time)
        else:
            self.time = np.arange(float(len(production)))
        self.eval_gradients()

    def eval_gradients(self):
        self.dp_dt = np.diff(self.production) / np.diff(self.time)        

    def matrix(self):
        return self.dp_dt.reshape(-1, 1)


class Normalizer:
    def __init__(self):
        self.features = skprep.MinMaxScaler(feature_range=(-1, 1))
        self.targets = skprep.MinMaxScaler(feature_range=(-1, 1))

    def normalize_features(self, features):
        return self.features.transform(features.matrix())

    def denormalize_features(self, feature_matrix):
        return self.features.inverse_transform(feature_matrix)

    def normalize_targets(self, targets):
        return self.targets.transform(targets.matrix())

    def denormalize_targets(self, target_matrix):
        return self.targets.inverse_transform(target_matrix)

    @staticmethod
    @makes_deep_copy    
    def fit(features, targets):
        normalizer = Normalizer()
        normalizer.features.fit(features.matrix())
        normalizer.targets.fit(targets.matrix())
        return normalizer

    def save(self, fname):
        pickle.dump(self, open(fname, "wb"))

    @staticmethod
    def load(fname):
        return pickle.load(open(fname, "rb"))


class ProgressBar:
    def __init__(self, num_iterations):
        self.fill = '█'
        self.length = 50
        self.decimals = 1
        self.num_iterations = num_iterations

    def __enter__(self):
        self.update()
        return self

    def __exit__(self, *args):
        print()

    def update(self, iteration=0, history=None):
        iteration = iteration + 1
        fraction = ("{0:." + str(self.decimals) + "f}").format(100.0*iteration/self.num_iterations)
        num_filled = int(self.length * iteration // self.num_iterations)
        bar = self.fill * num_filled + '-' * (self.length - num_filled)
        loss = 0.0
        if history:
            loss = history.history['loss'][-1]
        print('\rTraining |%s| %s%% complete, loss = %f.' % (bar, fraction, loss), end='\r')


def train(model, X, y, num_epochs, batch_size):
    with ProgressBar(num_epochs) as progress_bar:
        for i in range(num_epochs):
            history = model.fit(X, y, epochs=1, batch_size=batch_size, shuffle=False, verbose=0)
            model.reset_states()
            progress_bar.update(i, history)


@makes_deep_copy
def lstm(feature_matrix, target_matrix, batch_size, num_epochs, num_neurons):
    log.info('LSTM model with {0:d} neurons'.format(num_neurons))
    X, y = feature_matrix, target_matrix[:, 0]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = kem.Sequential()
    model.add(kel.LSTM(num_neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(kel.Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    train(model, X, y, num_epochs, batch_size)
    return model


NormalizerSeq = coll.namedtuple("NormalizerSeq", "time stage_delta production")


@makes_deep_copy
def lstmseq(time, production, stage):
    log.info('LSTM sequence model.')

    num_sequences = 1
    num_features = 2
    num_targets = 1
    num_timesteps = len(time)

    # sinc this is delta between [i] and [i-1], we loose the first row
    # and assume there is no stage change there, which wouldnt matter anyways
    # so for [0, 0, 1, 1] we produce a delta mapping of 
    #        [0, 0, 1, 0]
    stage_delta = np.zeros_like(stage)
    stage_delta[1:] = np.diff(stage)

    normalizer_stage_delta = skprep.MinMaxScaler(feature_range=(-1, 1))
    normalizer_production = skprep.MinMaxScaler(feature_range=(-1, 1))
    normalizer_time = skprep.MinMaxScaler(feature_range=(-1, 1))

    stage_delta_normalized = normalizer_stage_delta.fit_transform(stage_delta.reshape(-1, 1))
    production_normalized = normalizer_production.fit_transform(production.reshape(-1, 1))
    time_normalized = normalizer_time.fit_transform(time.reshape(-1, 1))
    
    X = np.zeros((num_sequences, num_timesteps, num_features))
    y = np.zeros((num_sequences, num_timesteps, num_targets))

    X[0, :, 0] = time_normalized[:, 0] # first feature is time
    X[0, :, 1] = stage_delta_normalized[:, 0] # second feature is state change
    y[0, :, 0] = production_normalized[:, 0] # only target is production
    
    log.info(X)
    log.info(y)
    
    # expected input data shape: (batch_size, timesteps, data_dim) 
    model = kem.Sequential()
    model.add(kel.LSTM(num_timesteps, input_shape=(num_timesteps, num_features), return_sequences=True))
    model.add(kel.TimeDistributed(kel.Dense(1)))
    model.compile(loss='mean_squared_error', optimizer='adam')

    """model = kem.Sequential()
    model.add(kel.LSTM(num_targets, 
      return_sequences=True, # same num as input, ie num_gradients_window
      input_shape=(num_timesteps, num_features)))  #  dQdt and stage_delta, returns a sequence (num_gradients_window) of vectors of dimension 2
    model.add(kel.TimeDistributed(kel.Dense(1)))
    model.compile(loss='mean_squared_error', optimizer='adam')"""
    
    model.summary()
    model.fit(X, y, batch_size=1, epochs=100)
    return model, NormalizerSeq(normalizer_time, normalizer_stage_delta, normalizer_production)


def predictseq(x, stage, normlizerseq, model):
    assert len(x) == len(stage)
    
    time = np.array(x)
    stage_delta = np.zeros_like(stage)
    stage_delta[1:] = np.diff(stage)
    num_timesteps = len(stage)
    num_features = 2

    stage_delta_normalized = normlizerseq.stage_delta.fit_transform(stage_delta.reshape(-1, 1))
    time_normalized = normlizerseq.time.fit_transform(time.reshape(-1, 1))

    X = np.zeros((1, num_timesteps, num_features))
    X[0, :, 0] = time_normalized[:, 0] # first feature is time
    X[0, :, 1] = stage_delta_normalized[:, 0] # second feature is state change

    yhat_normalized = model.predict(X)
    yhat = normlizerseq.production.inverse_transform(yhat_normalized[0])
    return x, yhat[:, 0]

    


def predict(y_0, stage, normalizer, model, time=None):
    yhat = [y_0]
    for i in range(1, len(stage)-1): 
        # input is first value, last discarded internally due to grad calc
        yprevious = yhat[-1]
        yinput = [yprevious, 0.0]
        stageinput = stage[i-1:i+1] # contains stage[i-1] and stage[i]
        features = Features(yinput, stageinput)
        features_normalized = normalizer.normalize_features(features)
        features_normalized_timeframe = features_normalized.reshape(features_normalized.shape[0], 
          1, features_normalized.shape[1])
        targets_normalized = model.predict(features_normalized_timeframe, batch_size=1)
        targets = normalizer.denormalize_targets(targets_normalized)
        dy_dt = targets[0, 0] 
        if time is None:
            time_delta = 1.0
        else:
            time_delta = time[i]-time[i-1]
        y_delta = time_delta * dy_dt
        yhat += [yprevious+y_delta]
    return np.array(yhat)


def load(fname):
    return kem.load_model(fname)


def save(model, fname):
    return model.save(fname)

