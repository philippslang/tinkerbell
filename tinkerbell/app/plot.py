import numpy as np
from matplotlib import pyplot as plt
import logging as log

plt.style.use('ggplot')

STYLEFALLBACK = {'linestyle': 'solid', 'linewidth': 2, 'alpha': 0.7}
TOMPLSTYLE = {'p': {'marker': 'x', 'linestyle': 'None'}, 'l': STYLEFALLBACK}


def plot(xyarraytuplesiterable, styles=[], labels=[], show=True):
    if not styles:
        styles = ['p'] * len(xyarraytuplesiterable)
    legend = True
    if not labels:
        legend = False
        labels = [None] * len(xyarraytuplesiterable)
    fig = plt.figure() 
    ax = fig.add_subplot(111)  
    for i, ((x, y), style, label) in enumerate(zip(xyarraytuplesiterable, styles, labels)):
        try:
            plotargs = TOMPLSTYLE[style]
        except:            
            series = label
            if not series:
                series = str(i)
            log.warn('Plot style \'{0}\' for series \'{1}\' not recognized, using fallback.'.format(style, label))
            plotargs = STYLEFALLBACK       
        ax.plot(x, y, **plotargs, label=label)
    if legend:
        plt.legend()
    if show:
        plt.show()