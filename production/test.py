from scipy import signal
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

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


df_a = pd.DataFrame({"utc": ["13:00","13:01","13:02"],
                   "lat": [1.1, 1.2, 1.3],
                   "lon": [-1.1, -2.1, -3.1],
                   'ne': [1e5, 2e5, 1e5],
                   "p_num": [1, 2, 2],
                   "sat": ["A", "A", "A"]})

df_b = pd.DataFrame({"utc": ["13:00","13:01","13:02"],
                   "lat": [2.1, 2.2, 2.3],
                   "lon": [-1.2, -2.2, -3.2],
                   'ne': [1e6, 2e6, 1e6],
                   "p_num": [1, 2, 2],
                   "sat": ["C", "C", "C"]})


df = pd.DataFrame({'date': ['2015-01-01', '2015-01-02', '2015-01-02', '2015-01-02', '2015-01-02','2015-01-02', '2015-01-02', '2015-01-02'],
                  'p_num': [1,2,2,2,5,5,5,5], 
                  #'Ne':[1e6, 1e5, 1e4, 5e6, 6e6,1e4, 5e6, 6e6],
                  'sg_smooth':[1,1,1,0,1,0,1,1],
                  'bub_id':[1,1,1,0,2,0,2,2]
                  })

#df_date = df.groupby()

#DUMMIES
import pandas as pd
 
df = pd.DataFrame.from_dict(
    {
        'heroes': ['Batman', 'Thor', 'Hulk', 'Spiderman', 'Flash'],
        'publisher': ['DC COMICS', 'Marvel', 'Marvel', 'Marvel', 'DC COMICS'],
        'Power': ['Medium', 'Strong', 'Strong', 'Medium', 'Low']
    }
)
 
print(df)

dummy = pd.get_dummies(df['publisher'])
print(dummy)
