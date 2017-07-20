import numpy as np
import pickle
import logging as log
import keras.models as kem
import keras.layers as kel
import sklearn.preprocessing as skprep


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

    @makes_deep_copy
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
def lstm(features, targets, batch_size, num_epochs, num_neurons):
    log.info('LSTM model with {0:d} neurons'.format(num_neurons))
    X, y = features, targets[:, 0]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = kem.Sequential()
    model.add(kel.LSTM(num_neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(kel.Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    train(model, X, y, num_epochs, batch_size)
    return model


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