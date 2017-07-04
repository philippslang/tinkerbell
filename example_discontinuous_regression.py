import point as pt
import factory as fc
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from keras.models import Sequential
from keras.layers import Dense
from example_discontinuous_tocsv import OFFSET_C, NUM_C, OFFSET_T, NUM_T, Y_MAX, K, FNAME, make_internal_knots
from example_discontinuous_fromcsv import make_spline


def model_deep():
    model = Sequential()
    afct = 'sigmoid'
    #afct = 'tanh'
    #afct = 'softmax'
    model.add(Dense(1, input_dim=1))
    model.add(Dense(NUM_T, activation=afct))
    model.add(Dense(NUM_C, activation=afct))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    np.random.seed(42)
    data = pd.read_csv(FNAME)
    dataset = data.values

    features = dataset[:,0]
    # coefficients only
    labels = dataset[:,NUM_T+1:]

    normalizer_features = MaxAbsScaler() #StandardScaler()
    features_normalized = normalizer_features.fit_transform(features)

    normalizer_labels = MaxAbsScaler() #StandardScaler()
    labels_normalized = normalizer_labels.fit_transform(labels)

    model = model_deep()
    model.fit(features_normalized, labels_normalized, epochs=300, batch_size=10)

    scores = model.evaluate(features_normalized, labels_normalized)
    model.save('data_demo/model.h5')
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


    features_test = [[Y_MAX/4], [Y_MAX/3], [Y_MAX/2], [Y_MAX/1.5]]
    features_test_normalized = normalizer_features.transform(features_test)
    labels_test_normalized = model.predict(features_test_normalized)
    labels_test = normalizer_labels.inverse_transform(labels_test_normalized)
    print(labels_test)
    spl_test = [make_spline(np.r_[make_internal_knots(features_test[i][0]), labels_test[i]]) for i in range(len(labels_test))]
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    xcoords_plot = np.linspace(0.0, Y_MAX, 200)
    for i in range(len(features_test)):
        ax.plot(xcoords_plot, spl_test[i](xcoords_plot), lw=3, label='{:.1f}'.format(features_test[i][0]))
    plt.legend()
    plt.show()



    

