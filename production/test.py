from scipy import signal
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

'''
def butter_highpass(low_cut, high_cut, fs, order=5):
    """
    Design band pass filter.

    Args:
        - low_cut  (float) : the low cutoff frequency of the filter.
        - high_cut (float) : the high cutoff frequency of the filter.
        - fs       (float) : the sampling rate.
        - order      (int) : order of the filter, by default defined to 5.
    """
    # calculate the Nyquist frequency
    nyq = 0.5 * fs

    # design filter
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = signal.butter(order, [low, high], btype='band')

    # returns the filter coefficients: numerator and denominator
    return b, a '''


#df = pd.DataFrame({'col1': [0,1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0,1,0,0,0,0,1]})


df = pd.DataFrame({'date': ['2015-01-01', '2015-01-02', '2015-01-02', '2015-01-02', '2015-01-02'],
                  'p_num': [1, 2, 2, 5, 5], 'Ne':[1e6, 1e5, 1e4, 5e6, 6e6] })


df_msssl = pd.DataFrame({"A": ["foo", "foo", "foo", "foo", "foo",
                         "bar", "bar", "bar", "bar"],
                   "B": ["one", "one", "one", "two", "two",
                         "one", "one", "two", "two"],
                   "C": ["small", "large", "large", "small",
                         "small", "large", "small", "small",
                         "large"],
                   "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
                   "E": [2, 4, 5, 5, 6, 6, 8, 9, 9]})

df_ibi = pd.DataFrame({"A": ["foo", "foo", "foo", "foo", "foo",
                         "bar", "bar", "bar", "bar"],
                   "B": ["one", "one", "one", "two", "two",
                         "one", "one", "two", "two"],
                   "C": ["small", "large", "large", "small",
                         "small", "large", "small", "small",
                         "large"],
                   "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
                   "E": [2, 4, 5, 5, 6, 6, 8, 9, 9]})