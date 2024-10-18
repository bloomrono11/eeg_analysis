import numpy as np
from scipy.stats import stats


# below are 12 classifications
def mean(x):
    return np.mean(x, axis=-1)


def std(x):
    return np.std(x, axis=-1)


def ptp(x):
    return np.ptp(x, axis=-1)


def var(x):
    return np.var(x, axis=-1)


def minim(x):
    return np.min(x, axis=-1)


def maxim(x):
    return np.max(x, axis=-1)


def argmin(x):
    return np.argmin(x, axis=-1)


def argmax(x):
    return np.argmax(x, axis=-1)


def rms(x):
    return np.sqrt(np.mean(np.square(x), axis=-1))


def abs_diff_signal(x):
    return np.sum(np.abs(np.diff(x, axis=-1)), axis=-1)


def skewness(x):
    return stats.skew(x, axis=-1)


def kurtosis(x):
    return stats.kurtosis(x, axis=-1)


# concatenate the above 12 classifications
def concat_features(x):
    return np.concatenate((mean(x), std(x), ptp(x), var(x), minim(x), maxim(x),
                           argmin(x), argmax(x), rms(x), abs_diff_signal(x),
                           skewness(x), kurtosis(x)), axis=-1)
