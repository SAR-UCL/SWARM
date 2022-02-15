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

print(df)

def ibi_mssl_epb_compare(df):

    df_new = df.groupby('p_num')
    df_arr = []
    for name, group in df_new:

        from sklearn.preprocessing import StandardScaler
        x_data = group[['Ne','p_num']]
        scaler = StandardScaler()
        scaler.fit(x_data) #compute mean for removal and std
        x_data = scaler.transform(x_data)
        ne_scale = [a[0] for a in x_data]
        group['Ne_scale'] = ne_scale

        def check_function(x,y):
            if x == 5e1 and y == -1:
            #if x == 1 and y == 1:
                return 'true pos'
            elif x == 0 and y == 0:
                return 'true neg'
            elif x == 1 and y == 0:
                return 'false neg'
            elif x == 0 and y == 1:
                return 'false pos'
            else:
                return 'no match'

        group['func_score'] = group.apply(lambda x: check_function(x['Ne'], 
                x['Ne_scale']), axis=1)

        score = group.groupby('func_score', as_index=False).size()

        date = group['date'].iloc[0]
        p_num = group['p_num'].iloc[0]
        #precision = score['size']
        precision = score.loc[score['func_score']=='no match', 'size']


        
        #precision = score['precision'].iloc[0]
        #precision = score.loc['precision']

        df = pd.DataFrame({'date':[date], 'p_num':p_num,  'precision':precision})
        df_arr.append(df)
        df = pd.concat(df_arr)

        #group['prescion'] = score.iloc[0]
        #print(precision)
        #print(date)

    #print(precision)
    #print(score)
    
    return df
    #return group

df = ibi_mssl_epb_compare(df)
print(df)