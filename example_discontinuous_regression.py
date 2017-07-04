import point as pt
import factory as fc
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from example_discontinuous_tocsv import OFFSET_C, NUM_C, OFFSET_T, NUM_T, Y_MAX, K, FNAME
from example_discontinuous_fromcsv import make_spline


def model_deep():
	model = Sequential()
	model.add(Dense(1, input_dim=1, activation='relu'))
	model.add(Dense(NUM_C, activation='relu'))
	model.add(Dense(NUM_C+NUM_T, activation='relu'))
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
	return model


if __name__ == '__main__':
    np.random.seed(42)
    data = pd.read_csv(FNAME)
    dataset = data.values

    features = dataset[:,0]
    labels = dataset[:,1:]

    normalizer_features = StandardScaler()
    features_normalized = normalizer_features.fit_transform(features)

    normalizer_labels = StandardScaler()
    labels_normalized = normalizer_labels.fit_transform(labels)

    model = model_deep()
    model.fit(features_normalized, labels_normalized, epochs=200, batch_size=10)

    scores = model.evaluate(features_normalized, labels_normalized)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


    features_test = [[Y_MAX/4], [Y_MAX/3], [Y_MAX/2], [Y_MAX/1.5]]
    features_test_normalized = normalizer_features.transform(features_test)
    labels_test_normalized = model.predict(features_test_normalized)
    labels_test = normalizer_labels.inverse_transform(labels_test_normalized)
    spl_test = [make_spline(labels_test_val) for labels_test_val in labels_test]
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    xcoords_plot = np.linspace(0.0, Y_MAX, 200)
    for i in range(len(features_test)):
        ax.plot(xcoords_plot, spl_test[i](xcoords_plot), lw=3, label='{:.1f}'.format(features_test[i][0]))
    plt.legend()
    plt.show()



    

