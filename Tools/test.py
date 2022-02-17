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

def demo_func(x):
    x = x - 1e6
    return x

#df['Ne_new'] = df.groupby('p_num')['Ne'].apply(demo_func)

#print(df)

arr = np.array([['2015-02-14', '00:00:23', 21.2924381044169]])

classified_df = pd.DataFrame(arr)
print(classified_df)
#print(arr)