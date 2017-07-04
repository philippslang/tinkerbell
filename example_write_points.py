import tinkerbell.domain.make as tbdmk
import tinkerbell.app.plot as tbapl
import tinkerbell.domain.point as tbdpt

y_i = 50
d = 0.1
x_max = 50.0
pts_exp_cont = tbdmk.points_exponential_discontinuous_decline_noisy(y_i, d, x_max, x_max, 1.0)
tbdpt.write_points('data_demo/points_01.json', pts_exp_cont)

x_disc = 20.0
pts_exp_discont = tbdmk.points_exponential_discontinuous_decline_noisy(y_i, d, x_max, x_disc)
tbdpt.write_points('data_demo/points_02.json', pts_exp_discont)

tbapl.plot_points([pts_exp_cont, pts_exp_discont])