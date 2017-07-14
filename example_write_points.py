import tinkerbell.app.make as tbamk
import tinkerbell.app.plot as tbapl
import tinkerbell.domain.point as tbdpt
import tinkerbell.app.plot as tbapl

y0 = 50.0
d = 0.1
xmax = y0
pmax = 6
xdisc = 2.0**pmax
pts_exp_cont, _ = tbamk.points_exponential_discontinuous_declinebase2_noisy(y0, d, pmax, xdisc)
tbdpt.write_points('data_demo/points_01.json', pts_exp_cont)

xdisc = 20.0
pts_exp_discont, _ = tbamk.points_exponential_discontinuous_declinebase2_noisy(y0, d, pmax, xdisc)
tbdpt.write_points('data_demo/points_02.json', pts_exp_discont)

xycoords_cont = tbdpt.point_coordinates(pts_exp_cont)
xycoords_disccont = tbdpt.point_coordinates(pts_exp_discont)
tbapl.plot([xycoords_cont, xycoords_disccont])
