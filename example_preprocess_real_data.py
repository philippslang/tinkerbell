import json, sys
import pandas as pd
import numpy as np
import logging as log
import tinkerbell.app.plot as tbapl
import tinkerbell.app.model as tbamd
import tinkerbell.app.make as tbamk
import tinkerbell.app.rcparams as tbarc


fname_json = 'data_demo/shale_fracflow00.json'
with open(fname_json) as f:
    data = json.load(f)

name_well = 'Evenson 1-18H LOT'
data_well = data[name_well]
time =  np.array(data_well['time'])
production =  np.array(data_well['production'])

stage = tbamk.detect_stages(time, production, num_stages_max=3, num_samples_window=5)

tbapl.plot([(time, production), (time, stage*np.mean(production))], 
  styles=['l', 'ls'], labels=['production', 'stage'])

#sys.exit()

features = tbamd.Features(production, stage)
targets = tbamd.Targets(production, time)

normalizer = tbamd.Normalizer.fit(features, targets) 
feature_matrix_normalized = normalizer.normalize_features(features)  
target_matrix_normalized = normalizer.normalize_targets(targets)

fname_model = tbarc.rcparams['shale.lstm_stage_well.fnamenmodel']
if 1:
    model = tbamd.lstm(feature_matrix_normalized, target_matrix_normalized, 1, 1500, 3)
    tbamd.save(model, fname_model)
else:
    model = tbamd.load(fname_model)

prediction = tbamd.predict(production[0], stage, normalizer, model, time)
time_prediction = time[:-1]

tbapl.plot([(time, production), (time_prediction, prediction)], 
  styles=['p', 'l'], labels=['ytrain', 'prediction'])
