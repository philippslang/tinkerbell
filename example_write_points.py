import tinkerbell.domain.make as tbdmk
import tinkerbell.app.plot as tbapl
import tinkerbell.domain.point as tbdpt

y0 = 50.0
d = 0.1
xmax = y0
pts_exp_cont = tbdmk.points_exponential_discontinuous_decline_noisy(y0, d, xmax, xmax, 1.0)
tbdpt.write_points('data_demo/points_01.json', pts_exp_cont)

xdisc = 20.0
pts_exp_discont = tbdmk.points_exponential_discontinuous_decline_noisy(y0, d, xmax, xdisc)
tbdpt.write_points('data_demo/points_02.json', pts_exp_discont)

tbapl.plot_points([pts_exp_cont, pts_exp_discont])