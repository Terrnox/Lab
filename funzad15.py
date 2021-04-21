import numpy as np


def y2(x):
    if x < 0:
        return np.sin(x)
    else:
        return np.sqrt(x)