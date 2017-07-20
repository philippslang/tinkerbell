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

name_well = 'Evenson'
name_well, time_refrac_predict = 'Austin', 50.0
#name_well = 'Hovda'
#name_well = 'Dantuhy'
inspect_only = False
version_model = 0
fit = False
num_stages_max = 2

data_well = data[name_well]
time =  np.array(data_well['time'])
production =  np.array(data_well['production'])

stage = tbamk.detect_stages(time, production, num_stages_max=num_stages_max, num_samples_window=10)

if inspect_only:
    tbapl.plot([(time, production), (time, stage*np.mean(production))], 
    styles=['l', 'ls'], labels=['production', 'stage'])
    sys.exit()


features = tbamd.Features(production, stage)
targets = tbamd.Targets(production, time)

normalizer = tbamd.Normalizer.fit(features, targets) 
feature_matrix_normalized = normalizer.normalize_features(features)  
target_matrix_normalized = normalizer.normalize_targets(targets)

fname_model = 'data_demo/model_' + name_well.lower() + '_v' + str(version_model) + '.h5'
if fit:
    model = tbamd.lstm(feature_matrix_normalized, target_matrix_normalized, 1, 2000, 3)
    tbamd.save(model, fname_model)
else:
    model = tbamd.load(fname_model)

time_predict = np.linspace(np.min(time), np.max(time), 75)
stage_predict = np.zeros_like(time_predict, dtype=np.int16)
stage_predict[time_predict > time_refrac_predict] = 1
production_predict = tbamd.predict(np.max(production), stage_predict, normalizer, model, time_predict)

time_predict = time_predict[:-1]
tbapl.plot([(time, production), (time_predict, production_predict)], 
  styles=['p', 'l'], labels=['ytrain', 'prediction'])
