import tinkerbell.app.plot as tbapl
import tinkerbell.domain.point as tbdpt
import tinkerbell.app.rcparams as tbarc
import pandas as pd
import numpy as np
import sklearn.preprocessing as preproc

"""from keras.models import Sequential, load_model
from keras.layers import Dense"""


data = pd.read_csv(tbarc.rcparams['shale.exp.csvtimefname'])

xcoords, ycoords = data['x'], data['y']

if 0:
    pts = tbdpt.from_coordinates(xcoords, ycoords)
    tbapl.plot_points([pts])

if 1:
    pass

