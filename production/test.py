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


# df = pd.concat([df_a, df_b],axis=0)
# print(df)

# sns.lineplot(data=df, x = 'utc',y='ne',hue='sat')
# plt.yscale('log')
# plt.tight_layout()
# plt.show()


path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Non-Flight Data/Analysis/Mar-22/data/sample/ml_pred_idea.csv'

df = pd.read_csv(path)
#print(df)

df_piv = df.pivot_table(values='diff', index='day', columns='month', dropna = False)
print(df_piv)

#fig, ax = plt.figure(figsize = (6.5,6.5))
fig, ax = plt.subplots(figsize = (6.5,6.5))
cmap = "PiYG" #differentiation
#cmap = "YlGnBu" #individual days
sns.heatmap(df_piv, annot=True, linewidths=0.1,
            fmt=".0f", cmap=cmap, center=0)

plt.title(f'EPB Predictions vs. Actual. \n where 0 = match, > 0 false positives, < 0 false negatives'
            '\n the perfect score is 0 on all dates ', fontsize=10.5)

ax.text(0.5, 0.5, 'Sample data', transform=ax.transAxes,
        fontsize=40, color='gray', alpha=0.3,
        ha='center', va='center', rotation='30')

#plt.title(f'Number of EPB events in {date}, \n (MSSL Classifier)', fontsize=10.5)
plt.xlabel(' ')
plt.ylabel(' ')

#ax.figure.axes[-1].set_ylabel(f'{log} counts per sec', size=9.5)
plt.yticks(rotation = 0)

plt.tight_layout()
plt.show()