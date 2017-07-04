import point as pt
import factory as fc
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
from example_discontinuous_tocsv import OFFSET_C, NUM_C, OFFSET_T, NUM_T, Y_MAX, K
from scipy.interpolate import BSpline


def make_spline(labels):
    c = np.r_[labels[NUM_T:], [0., 0., 0.]]
    t = np.r_[[0.]*3, labels[:NUM_T], [Y_MAX]*3]    
    return BSpline(t, c, K)


if __name__ == '__main__':
    data = pd.read_csv('data_demo/discontinuous00.csv')

    for row in data.itertuples():
        spl = make_spline(row[2:]) # rm index and feature
        
        fig = plt.figure() 
        ax = fig.add_subplot(111)
        xcoords_plot = np.linspace(0.0, Y_MAX, 200)
        ax.plot(xcoords_plot, spl(xcoords_plot), 'g-', lw=3)
        plt.show()
        break

    

