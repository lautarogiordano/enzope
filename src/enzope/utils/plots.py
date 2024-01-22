import numpy as np
from matplotlib import pyplot as plt

def distribution(w, bins=50, title=None, figsize=(8, 6), **kwargs):
    """Plot the distribution of a vector of weights."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.hist(w, bins=bins, **kwargs)
    ax.set_xlabel('weight')
    ax.set_ylabel('frequency')
    if title is not None:
        ax.set_title(title)
    return fig
