import logging as log
import keras.models as kem
import keras.layers as kel


def printProgressBar (iteration, total, prefix='Training:', suffix='complete', decimals=1, length=50, fill='█', history=None):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    if not history:
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    else:
        print('\r%s |%s| %s%% %s, loss = %f.' % (prefix, bar, percent, suffix, history.history['loss'][-1]), end='\r')
    # Print New Line on Complete
    if iteration == total: 
        print()


def train(model, X, y, num_epochs, batch_size):
    print('')
    printProgressBar(0, num_epochs)
    for i in range(num_epochs):
        history = model.fit(X, y, epochs=1, batch_size=batch_size, shuffle=False, verbose=0)
        model.reset_states()
        printProgressBar(i+1, num_epochs, history=history)


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