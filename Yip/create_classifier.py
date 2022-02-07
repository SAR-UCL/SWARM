
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
pd.set_option('display.max_rows', 10) #or 10 or None


path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Non-Flight Data/Analysis/Jan-22/data/systematic/train_set/'
load_data = path + 'train_set.csv'
df = pd.read_csv(load_data)

test_id = 3
df = df[df['id']==test_id]
#print(df)

def covar_corr(df):
    df['ne_pot_corr'] = df['Ne'].rolling(window=10).corr(df['Ti'])
    df['ne_pot_covar'] = df['Ne'].rolling(window=10).cov(df['Ti'])


    def covar_check(covar):
        if covar > 7e5:
            return 1
        else:
            return 0

    df['covar_epb'] = df['ne_pot_covar'].apply(covar_check)

    return df

df = covar_corr(df)

def calc_ROC_pt(df):
    #Rate of change cm/s or k/s or pot/s
    pc_df = df[['Ne','pot','Ti']].pct_change(periods=1) #change in seconds
    pc_df = pc_df.rename(columns = {"Ne":"Ne_c", "pot":"pot_c","Ti":"Ti_c"}) 
    df = pd.concat([df, pc_df], axis=1)

    df = df.dropna()

    return df


def stddev_window(df):

    df = calc_ROC_pt(df)

    std_df = df[['Ne_c','pot_c','Ti_c']].rolling(5).std()
    #std_df = df[['Ne_c','pot_c']].rolling(20, win_type='gaussian').sum(std=3)
    std_df = std_df.rename(columns = {"Ne_c":"Ne_std", "pot_c":"pot_std","Ti_c":"Ti_std"}) 
    df = pd.concat([df,std_df], axis = 1)

    #df = df.drop(columns=['Ne_c','pot_c'])

    def stddev_ne(x):
        if x > 0.1:
            return 1
        else:
            return 0

    def stddev_pot(x):
        if x > 0.01:
            return 1
        else:
            return 0

    def stddev_ti(x):
        if x > 0.1:
            return 1
        else:
            return 0


    df['std_ne'] = df['Ne_std'].apply(stddev_ne)
    df['std_pot'] = df['pot_std'].apply(stddev_pot)
    df['std_ti'] = df['Ti_std'].apply(stddev_ti)

    df = df.dropna()

    return df

df = stddev_window(df)


def gauss_window(df):

    #std_df = df[['Ne_c','pot_c']].rolling(10).std()
    std_df = df[['Ne_c','pot_c']].rolling(10).min()
    #std_df = df[['Ne_c','pot_c']].rolling(5, win_type='gaussian').sum(std=2)
    std_df = std_df.rename(columns = {"Ne_c":"Ne_gau", "pot_c":"pot_gau"}) 
    df = pd.concat([df,std_df], axis = 1)

    #df = df.drop(columns=['Ne_c','pot_c'])

    def stddev_check(x):
        if -0.001 <= x < 0:
            return 1
        else:
            return 0

    df['gauss'] = df['Ne_gau'].apply(stddev_check)

    df = df.dropna()

    return df

#df = gauss_window(df)



def savitzky_golay(y):
    from math import factorial
    window_size = np.abs(np.int(10))
    order = np.abs(np.int(10))
    order_range = range(3+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[0] * 1**0 * factorial(0)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

#print(df)
#df['func_score'] = df.apply(lambda x: savitzky_golay(x['Ne']), axis=1)


def ne_pot_combine(ne, pot):
    if ne and pot == 1:
        return 1
    else:
        return 0

#df['ne_pot'] = df.apply(lambda x: ne_pot_combine(x['std_ne'], 
#         x['std_pot']), axis=1)
         

def check_function(x,y):
    if x == 1 and y == 1:
        return 'true pos'
    elif x == 0 and y == 0:
        return 'true neg'
    elif x == 1 and y == 0:
        return 'false neg'
    elif x == 0 and y == 1:
        return 'false pos'
    else:
        return 0 

df['func_score'] = df.apply(lambda x: check_function(x['epb_gt'], 
         x['covar_epb']), axis=1)

scores = df.groupby('func_score').size()
#print(scores)

if test_id == 2 or test_id == 6 or test_id == 7 or test_id == 3:
#if test_id == 10:
    print('Scores',scores)
    precision = 0 / (1 + 0)
    recall = 0 / (0 + scores.iloc[0])
    #print(recall)
    
else:
    precision = scores.iloc[3] / (scores.iloc[3] + scores.iloc[1])
    recall = scores.iloc[3] / (scores.iloc[3] + scores.iloc[0])
    f1 = 2*((precision*recall)/(precision+recall))
    precision = "{:.2f}".format(precision) 
    recall = "{:.2f}".format(recall) 
    f1 = "{:.2f}".format(f1)

    print('Scores',scores)
    print ('Precision:', precision)
    print ('Recall:', recall)
    print('F1:',f1)

print(df)

def plotNoStdDev(df):
        
        figs, axs = plt.subplots(ncols=1, nrows=5, figsize=(8,5), 
        dpi=90, sharex=True) #3.5 for single, #5.5 for double
        axs = axs.flatten()

        x = 'lat'
        hue = 's_id'

        ax0y = 'Ne'
        sns.lineplot(ax = axs[0], data = df, x = x, y =ax0y, 
                palette = 'bone',hue = hue, legend=False)

        ax1y = 'epb_gt'
        sns.lineplot(ax = axs[1], data = df, x =x, y =ax1y,
                palette = 'Set1', hue = hue, legend=False)

        ax2y = 'ne_pot_covar'
        sns.lineplot(ax = axs[2], data = df, x = x, y =ax2y, 
                palette = 'Set1', hue = hue, legend = False)

        ax3y = 'covar_epb'
        sns.lineplot(ax = axs[3], data = df, x = x, y =ax3y, 
                palette = 'Set2', hue = hue, legend = False)

        ax4y = 'ne_pot_corr'
        sns.lineplot(ax = axs[4], data = df, x = x, y =ax4y, 
                palette = 'Set2', hue = hue, legend = False)

        #axs3 = axs[2].twinx()
        #sns.lineplot(ax = axs3, data = df, x = x, y ='epb_gt', 
        #        palette = 'Set2', hue = hue, legend = False)

        date_s = df['date'].iloc[0]
        date_e = df['date'].iloc[-1]
        utc_s = df['utc'].iloc[0]
        utc_e = df['utc'].iloc[-1]
      
        lat_s = df['lat'].iloc[0]
        lat_e = df['lat'].iloc[-1]

        epb_len = (lat_s - lat_e) * 110
        epb_len = "{:.0f}".format(epb_len)
        
        title = 'EPB Classifier testing. EPB: '

        axs[0].set_title(f'{title}: from {date_s} at {utc_s} to {date_e} at {utc_e}'
               #f'\n Precision: {precision}, Recall: {recall}, F1: {f1}' 
                ,fontsize = 11)

        den = r'cm$^{-3}$'
        axs[0].set_ylabel(f'{ax0y}')
        axs[0].tick_params(bottom = False)
        
        axs[1].set_ylabel(f'{ax1y}')
        axs[1].tick_params(bottom = False)
        #axs[2].set_yscale('log')
        
        axs[2].set_ylabel(f'{ax2y}')
        axs[2].tick_params(bottom = False)

        axs[3].set_ylabel(f'{ax3y}')
        axs[3].tick_params(bottom = False)

        axs[4].set_ylabel(f'{ax3y}')
        axs[4].tick_params(bottom = False)

        #axs3.set_ylabel('Actual (Green)')

        ax = plt.gca()
        ax.invert_xaxis()

        plt.tight_layout()
        plt.show()

plotNoStdDev(df)