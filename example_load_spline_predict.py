"""
Multilayer perceptron regression model
"""

import tinkerbell.app.plot as tbapl
import tinkerbell.domain.make as tbdmk
import tinkerbell.domain.curve as tbdcv
import tinkerbell.domain.point as tbdpt
import tinkerbell.app.make as tbamk
import tinkerbell.app.rcparams as tbarc
import pandas as pd
import numpy as np
import example_regress_on_spline_training_set
import example_generate_spline_training_set
from keras.models import load_model
import pickle

fname_model = example_regress_on_spline_training_set.FNAME_MODEL
model = load_model(fname_model)
normalizer_labels = pickle.load(open(example_regress_on_spline_training_set.FNAME_ONORM, "rb"))
normalizer_features = pickle.load(open(example_regress_on_spline_training_set.FNAME_INORM, "rb"))

y0_mean = tbarc.rcparams['shale.exp.y0_mean']
dx = tbarc.rcparams['shale.exp.dx']
k = tbarc.rcparams['shale.exp.k']
d = example_generate_spline_training_set.D
pmax = example_generate_spline_training_set.PMAX
xmin = 0.0
xmax = 2.0**pmax
features_test = [[xmax/4], [xmax/3], [xmax/2], [xmax/1.5]]
features_test_normalized = normalizer_features.transform(features_test)
labels_test_normalized = model.predict(features_test_normalized)
labels_test = normalizer_labels.inverse_transform(labels_test_normalized)

internal_knots_tests = [tbamk.knots_internal_four_heavy_right(xdisc[0], xmax, dx) for xdisc in features_test]
knots_tests = [tbdmk.knots_from_internal_knots(k, internal_knots, xmin, xmax) for internal_knots in internal_knots_tests]
curves_test = [tbdcv.Curve(knots_tests[i], labels_test[i], k) for i in range(len(features_test))]
xycoords_crvtest = [crv.xycoordinates() for crv in curves_test]
styles = ['l' for _ in curves_test]
labels = ['yhat {:.1f}'.format(features_test[i][0]) for i in range(len(features_test))]

for xdiscc in features_test:
    xdisc = xdiscc[0]
    pts, _ = tbamk.points_exponential_discontinuous_declinebase2_noisy(y0_mean, d, pmax, xdisc)
    xycoords_crvtest.append(tbdpt.point_coordinates(pts))
    styles.append('p')
    labels.append('yblind {:.1f}'.format(xdisc))
    

print(len(xycoords_crvtest))
tbapl.plot(xycoords_crvtest, styles=styles, labels=labels)

