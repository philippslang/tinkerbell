import numpy as np
from matplotlib import pyplot as plt
import logging as log

plt.style.use('ggplot')

STYLEFALLBACK = {'linestyle': 'solid', 'linewidth': 2, 'alpha': 0.7}
TOMPLSTYLE = {'p': {'marker': 'x', 'linestyle': 'None'}, 'l': STYLEFALLBACK,
  'ls': {'linestyle': 'dashed', 'linewidth': 2, 'alpha': 0.7},
  'iy': {'marker': '>', 'linestyle': 'None', 'color': 'k', 'markerfacecolor': 'None'},
  'ix': {'marker': '^', 'linestyle': 'None', 'color': 'k', 'markerfacecolor': 'None'}}


def render(ax, xyarraytuplesiterable, styles=[], labels=[], lim=((None, None), (None, None))):
    if not styles:
        styles = ['p'] * len(xyarraytuplesiterable)
    legend = True
    if not labels:
        legend = False
        labels = [None] * len(xyarraytuplesiterable)
    
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
    ax.set_xlim(lim[0][0], lim[0][1])
    ax.set_ylim(lim[1][0], lim[1][1])
    if legend:
        ax.legend()


def plot(xyarraytuplesiterable, styles=[], labels=[], show=True, lim=((None, None), (None, None))):
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    render(ax, xyarraytuplesiterable, styles, labels, lim=lim)
    if show:
        plt.show()