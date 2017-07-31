import tinkerbell.domain.point as tbdpt
import tinkerbell.domain.make as tbdmk
import tinkerbell.app.make as tbamk
import tinkerbell.app.plot as tbapl
import numpy as np

pts = tbdpt.read_points('data_demo/points_02.json')

xmax = 50.0
xdisc = 20.0
dx = xmax/50
k = 2
t = tbamk.knots_internal_four_heavy_right(xdisc, xmax, dx)
crv = tbdmk.curve_lsq_fixed_knots(pts, t, k)

xcoordscrv, ycoordscrv = crv.xycoordinates()
xcoordspts, ycoordspts = tbdpt.point_coordinates(pts) 


#tbapl.plot([(xcoordspts, ycoordspts)], ['p'], hide_labels=True, save_as='img/curve_input.png', ylabel='production', xlabel='time')

if 0:
    stage = np.zeros_like(xcoordspts)
    stage[36:] = 1
    tbapl.plot([(xcoordspts, ycoordspts), (xcoordspts, stage*np.mean(ycoordspts))], styles=['p', 'ls'], 
    labels=['production', 'stage'], hide_labels=True,  ylabel='production', xlabel='time', secylabel='stage', save_as='img/curve_stage.png')

tbapl.plot([(xcoordspts, ycoordspts), (xcoordscrv, ycoordscrv)], ['p', 'l'], hide_labels=True,
  ylabel='production', xlabel='time', save_as='img/curve_fit.png')
