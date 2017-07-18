import logging as log
import keras.models as kem
import keras.layers as kel
import numpy as np


class Features:
    def __init__(self, production, stage):
        self.production = np.copy(production)
        self.stage = np.copy(stage)
        self.eval_deltas()

    def eval_deltas(self):
        self.dproduction_dt = np.diff(self.production)
        self.dstage_dstep = np.diff(self.stage)


class ProgressBar:
    def __init__(self, num_iterations):
        self.fill = '█'
        self.length = 50
        self.decimals = 1
        self.num_iterations = num_iterations
        self.leading_newline = True

    def __enter__(self):
        if self.leading_newline:
            print()
        self.update()
        return self

    def __exit__(self, *args):
        print()

    def update(self, iteration=0, history=None):
        fraction = ("{0:." + str(self.decimals) + "f}").format(100 * (iteration/self.num_iterations)))
        num_filled = int(self.length * iteration // self.num_iterations)
        bar = self.fill * num_filled + '-' * (self.length - num_filled)
        print('\rTraining |%s| %s%% complete, loss = %f.' % (bar, fraction, history.history['loss']), end='\r')


def train(model, X, y, num_epochs, batch_size):
    with ProgressBar(num_epochs) as progress_bar:
        for i in range(num_epochs):
            history = model.fit(X, y, epochs=1, batch_size=batch_size, shuffle=False, verbose=0)
            model.reset_states()
            progress_bar.update(i, history)


def lstm(features, labels, batch_size, num_epochs, num_neurons):
    log.info('LSTM model with {0:d} neurons'.format(num_neurons))
    X, y = features, labels[:, 0]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = kem.Sequential()
    model.add(kel.LSTM(num_neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(kel.Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    train(model, X, y, num_epochs, batch_size)
    return model


def load(fname):
    return kem.load_model(fname)


def save(model, fname):
    return model.save(fname)