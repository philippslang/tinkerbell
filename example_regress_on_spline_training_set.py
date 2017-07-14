"""
Multilayer perceptron regression model
"""

import tinkerbell.app.plot as tbapl
import tinkerbell.domain.make as tbdmk
import tinkerbell.domain.curve as tbdcv
import tinkerbell.app.make as tbamk
import tinkerbell.app.rcparams as tbarc
import pandas as pd
import numpy as np
import sklearn.preprocessing as preproc

from keras.models import Sequential, load_model
from keras.layers import Dense

k = tbarc.rcparams['shale.exp.k']
num_knots_internal = tbarc.rcparams['shale.exp.num_knots_internal']
num_knots = tbdmk.num_knots_curve_lsq(k, num_knots_internal)

def model_deep():
    model = Sequential()
    afct = 'sigmoid'
    #afct = 'tanh'
    #afct = 'softmax'
    model.add(Dense(1, input_dim=1))
    model.add(Dense(num_knots_internal, activation=afct))
    model.add(Dense(num_knots, activation=afct))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model

data = pd.read_csv(tbarc.rcparams['shale.exp.csvsplinefname'])

features = data['xdisc']
features = features.values.reshape(-1, 1) # single feature for now
normalizer_features = preproc.MinMaxScaler() 
features_normalized = normalizer_features.fit_transform(features)

labels_header = tbdcv.flat_header_coefficients(num_knots)
labels = data[labels_header]
normalizer_labels = preproc.MinMaxScaler() 
labels_normalized = normalizer_labels.fit_transform(labels)

fname_model = 'data_demo/model.h5'
if 0:
    model = model_deep()
    model.fit(features_normalized, labels_normalized, epochs=400, batch_size=30)
    scores = model.evaluate(features_normalized, labels_normalized)
    model.save(fname_model)
else:
    model = load_model(fname_model)

y0_mean = tbarc.rcparams['shale.exp.y0_mean']
dx = tbarc.rcparams['shale.exp.dx']
xmin = 0.0
xmax = y0_mean
features_test = [[xmax/4], [xmax/3], [xmax/2], [xmax/1.5]]
features_test_normalized = normalizer_features.transform(features_test)
labels_test_normalized = model.predict(features_test_normalized)
labels_test = normalizer_labels.inverse_transform(labels_test_normalized)

internal_knots_tests = [tbamk.knots_internal_four_heavy_right(xdisc[0], xmax, dx) for xdisc in features_test]
knots_tests = [tbdmk.knots_from_internal_knots(k, internal_knots, xmin, xmax) for internal_knots in internal_knots_tests]
curves_test = [tbdcv.Curve(knots_tests[i], labels_test[i], k) for i in range(len(features_test))]
xycoords_crvtest = [crv.xycoordinates() for crv in curves_test]
styles = ['l' for _ in curves_test]
tbapl.plot(xycoords_crvtest, styles=styles, labels=['{:.1f}'.format(features_test[i][0]) for i in range(len(features_test))])
