"""
Recurrent neural network model example with one-stage production data.
"""
import sys
import pandas as pd
import numpy as np
import logging as log
import pickle
import tinkerbell.app.plot as tbapl
import tinkerbell.app.model as tbamd
import tinkerbell.app.make as tbamk
import tinkerbell.app.rcparams as tbarc
import tinkerbell.domain.point as tbdpt


def do_the_thing():
    fname_csv = tbarc.rcparams['shale.lstm.fnamecsv']
    fname_csv = tbarc.rcparams['shale.lstm_stage.fnamecsv']
    name_dataset = fname_csv[10:-4]
    log.info(name_dataset)
    series = pd.read_csv(fname_csv)

    y = series['y'].values
    x = series['x'].values
    stage = series['stage'].values

    if 0:
        tbapl.plot([(x, y)], styles=['p'], 
          labels=['production'], hide_labels=True,  ylabel='production', xlabel='time', secylabel='stage', 
          secxyarraytuplesiterable=[(x, stage*np.mean(y))], secstyles=['lstage'], seclabels=['stage'], 
          save_as='img/lstm_pts_stage.png', secylim=(None, 50))
        sys.exit()
 
    fname_model =  tbarc.rcparams['shale.lstm.sequence.win.fnamemodel'][:-3] + '_' + name_dataset + '.h5'
    fname_normalizer = tbarc.rcparams['shale.lstm.sequence.win.fnamenorm'][:-3] + '_' + name_dataset + '.h5'
    num_timesteps = 2
    num_units = 3
    num_epochs = 100
    offset_forecast = 1#num_timesteps
    if 0:
        model, normalizer = tbamd.lstmseqwin(y, stage, num_epochs, num_timesteps, 
          num_units, offset_forecast)
        tbamd.save(model, fname_model)
        pickle.dump(normalizer, open(fname_normalizer, 'wb'))
    else:
        model = tbamd.load(fname_model)
        normalizer = pickle.load(open(fname_normalizer, 'rb'))

    if 1:
        ypred = tbamd.predictseqwin(y[:num_timesteps], stage, normalizer, model, offset_forecast)
        xpred = x[:len(ypred)]
        lim = ((-5, 95), (-2, 55))
        tbapl.plot([(x[:3], y[:3])], styles=['p'], labels=['initial production'],  ylabel='production', xlabel='time',
          hide_labels=True, save_as='img/lstm_pts_stage_pred00.png', lim=lim, secylabel='stage', 
          secxyarraytuplesiterable=[(x, stage*np.mean(y))], secstyles=['lstage'], seclabels=['planned stage'], 
          secylim=(None, 50))
        
        tbapl.plot([(x[:3], y[:3]), (xpred, ypred)], styles=['p', 'lblack'], labels=['initial production', 'predicted production'],  ylabel='production', xlabel='time',
          hide_labels=True, save_as='img/lstm_pts_stage_pred01.png', lim=lim, secylabel='stage', 
          secxyarraytuplesiterable=[(x, stage*np.mean(y))], secstyles=['lstage'], seclabels=['planned stage'], 
          secylim=(None, 50))

        tbapl.plot([(x, y), (xpred, ypred)], styles=['p', 'lblack'], labels=['training production', 'predicted production'],  ylabel='production', xlabel='time',
          hide_labels=True, save_as='img/lstm_pts_stage_pred02.png', lim=lim, secylabel='stage', 
          secxyarraytuplesiterable=[(x, stage*np.mean(y))], secstyles=['lstage'], seclabels=['reference stage'], 
          secylim=(None, 50))

    if 1:
        stage1 = np.zeros_like(stage)
        ixdisc = 25
        stage1[ixdisc:] = 1
        ypred = tbamd.predictseqwin(y[:num_timesteps], stage1, normalizer, model, offset_forecast)
        xpred = x[:len(ypred)]
        lim = ((-5, 95), (-2, 55))
        tbapl.plot([(x[:3], y[:3])], styles=['p'], labels=['initial production'],  ylabel='production', xlabel='time',
          hide_labels=True, save_as='img/lstm_pts_stage_pred10.png', lim=lim, secylabel='stage', 
          secxyarraytuplesiterable=[(x, stage1*np.mean(y))], secstyles=['lstage'], seclabels=['planned stage'], 
          secylim=(None, 50))
        
        #sys.exit()
        tbapl.plot([(x[:3], y[:3]), (xpred, ypred)], styles=['p', 'lblack'], labels=['initial production', 'predicted production'],  ylabel='production', xlabel='time',
          hide_labels=True, save_as='img/lstm_pts_stage_pred11.png', lim=lim, secylabel='stage', 
          secxyarraytuplesiterable=[(x, stage1*np.mean(y))], secstyles=['lstage'], seclabels=['planned stage'], 
          secylim=(None, 50))

        tbapl.plot([(x, y), (xpred, ypred)], styles=['p', 'lblack'], labels=['training production', 'predicted production'],  ylabel='production', xlabel='time',
          hide_labels=True, save_as='img/lstm_pts_stage_pred12.png', lim=lim, secylabel='stage', 
          secxyarraytuplesiterable=[(x, stage1*np.mean(y))], secstyles=['lstage'], seclabels=['planned stage'], 
          secylim=(None, 50))

        y0 = tbarc.rcparams['shale.lstm.y0_mean']
        d = tbarc.rcparams['shale.lstm_stage.d']
        xmax = tbarc.rcparams['shale.lstm_stage.xmax']
        num_points = tbarc.rcparams['shale.lstm_stage.num_points']
        pts, _ = tbamk.points_exponential_discontinuous_declinelinear_noisy(y0, d, xmax, xpred[ixdisc], num=num_points)
        tbapl.plot([tbdpt.point_coordinates(pts), (xpred, ypred)], styles=['p', 'lblack'], labels=['reference production', 'predicted production'],  ylabel='production', xlabel='time',
          hide_labels=True, save_as='img/lstm_pts_stage_pred12.png', lim=lim, secylabel='stage', 
          secxyarraytuplesiterable=[(x, stage1*np.mean(y))], secstyles=['lstage'], seclabels=['planned stage'], 
          secylim=(None, 50))
        
    if 1:
        stage1 = np.zeros_like(stage)
        ixdisc = 42
        stage1[ixdisc:] = 1
        ypred = tbamd.predictseqwin(y[:num_timesteps], stage1, normalizer, model, offset_forecast)
        xpred = x[:len(ypred)]
        lim = ((-5, 95), (-2, 55))
        tbapl.plot([(x[:3], y[:3])], styles=['p'], labels=['initial production'],  ylabel='production', xlabel='time',
          hide_labels=True, save_as='img/lstm_pts_stage_pred20.png', lim=lim, secylabel='stage', 
          secxyarraytuplesiterable=[(x, stage1*np.mean(y))], secstyles=['lstage'], seclabels=['planned stage'], 
          secylim=(None, 50))
        
        #sys.exit()
        tbapl.plot([(x[:3], y[:3]), (xpred, ypred)], styles=['p', 'lblack'], labels=['initial production', 'predicted production'],  ylabel='production', xlabel='time',
          hide_labels=True, save_as='img/lstm_pts_stage_pred21.png', lim=lim, secylabel='stage', 
          secxyarraytuplesiterable=[(x, stage1*np.mean(y))], secstyles=['lstage'], seclabels=['planned stage'], 
          secylim=(None, 50))

        y0 = tbarc.rcparams['shale.lstm.y0_mean']
        d = tbarc.rcparams['shale.lstm_stage.d']
        xmax = tbarc.rcparams['shale.lstm_stage.xmax']
        num_points = tbarc.rcparams['shale.lstm_stage.num_points']
        pts, _ = tbamk.points_exponential_discontinuous_declinelinear_noisy(y0, d, xmax, xpred[ixdisc], num=num_points)
        tbapl.plot([tbdpt.point_coordinates(pts), (xpred, ypred)], styles=['p', 'lblack'], labels=['reference production', 'predicted production'],  ylabel='production', xlabel='time',
          hide_labels=True, save_as='img/lstm_pts_stage_pred22.png', lim=lim, secylabel='stage', 
          secxyarraytuplesiterable=[(x, stage1*np.mean(y))], secstyles=['lstage'], seclabels=['planned stage'], 
          secylim=(None, 50))


if __name__ == '__main__':
    #log.basicConfig(filename='debug01.log', level=log.DEBUG, filemode='w')
    do_the_thing()
