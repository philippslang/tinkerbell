import numpy as np
from matplotlib import pyplot as plt
import logging as log

plt.style.use('ggplot')

STYLEFALLBACK = {'linestyle': 'solid', 'linewidth': 2, 'alpha': 0.7}
TOMPLSTYLE = {'p': {'marker': 'x', 'linestyle': 'None'}, 'l': STYLEFALLBACK,
  'ls': {'linestyle': 'dashed', 'linewidth': 2, 'alpha': 0.7},
  'lstage': {'linestyle': 'solid', 'linewidth': 2, 'alpha': 0.3, 'color': 'blue'},
  'iy': {'marker': '>', 'linestyle': 'None', 'color': 'k', 'markerfacecolor': 'None'},
  'ix': {'marker': '^', 'linestyle': 'None', 'color': 'k', 'markerfacecolor': 'None'}}


def render(ax, xyarraytuplesiterable, styles=[], labels=[], lim=((None, None), (None, None))):
    if not styles:
        styles = ['p'] * len(xyarraytuplesiterable)
    if not labels:
        labels = [None] * len(xyarraytuplesiterable)
    handlesret, labelsret = [], []
    for i, ((x, y), style, label) in enumerate(zip(xyarraytuplesiterable, styles, labels)):
        try:
            plotargs = TOMPLSTYLE[style]
        except:            
            series = label
            if not series:
                series = str(i)
            log.warn('Plot style \'{0}\' for series \'{1}\' not recognized, using fallback.'.format(style, label))
            plotargs = STYLEFALLBACK       
        h = ax.plot(x, y, **plotargs, label=label)
        handlesret += h
        labelsret += [label]
    ax.set_xlim(lim[0][0], lim[0][1])
    ax.set_ylim(lim[1][0], lim[1][1])
    return handlesret, labelsret


def plot(xyarraytuplesiterable, styles=[], labels=[], show=True, lim=((None, None), (None, None)), 
         save_as='', secxyarraytuplesiterable=[], seclabels=[], secstyles=[], hide_labels=False,
         xlabel='', ylabel='', secylabel=''):
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    handelsleg, labelsleg = [], []
    hret, lret = render(ax, xyarraytuplesiterable, styles, labels, lim=lim)
    handelsleg += hret
    labelsleg += lret

    if hide_labels:
        ax.set_xticklabels(['']*len(ax.get_xticklabels()))
        ax.set_yticklabels(['']*len(ax.get_yticklabels()))

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    if secxyarraytuplesiterable:
        axsec = ax.twinx()
        hret, lret = render(axsec, secxyarraytuplesiterable, secstyles, seclabels)
        handelsleg += hret
        labelsleg += lret
        # aligning ticks
        ax.set_yticks(np.linspace(ax.get_ybound()[0], ax.get_ybound()[1], 5))
        axsec.set_yticks(np.linspace(axsec.get_ybound()[0], axsec.get_ybound()[1], 5))

        if hide_labels:
            axsec.set_xticklabels(['']*len(axsec.get_xticklabels()))
            axsec.set_yticklabels(['']*len(axsec.get_yticklabels()))

        if secylabel:
            axsec.set_ylabel(secylabel)
    if labels:
        fig.legend(handelsleg, labelsleg)
    if save_as:
        plt.savefig(save_as, dpi=300)
    if show:
        plt.show()