
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
pd.set_option('display.max_rows', 10) #or 10 or None


path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Non-Flight Data/Analysis/Mar-22/data/train-test-sg/'
load_data = path + 'train-test_set.csv'
df = pd.read_csv(load_data)

df = df[['date','utc','mlt','lat','long','s_id','pass','Ne','pot','id','epb_gt']]

test_id = 9
df = df[df['id']==test_id]
#print(df)


def savitzky_golay(df):

    from scipy.signal import savgol_filter
    
    df['pot_savgol'] = savgol_filter(df['pot'], window_length=23,
        polyorder = 3) #Ne do not change    
    df['pot_resid'] = df['pot'] - df['pot_savgol']
    df = df.dropna()
    return df
df = savitzky_golay(df)

def savgol_threshold(x):
    if x > 0.001 or x < -0.001: #non-norm
    #if x > 0.05 or x < -0.05: #norm
        return 1
    else:
        return 0

df['sg_epb'] = df.apply(lambda x: savgol_threshold(x['pot_resid']), axis=1)

def smooth_savgol(x, y, z):
    if x == 1:
        return 1
    elif y + z == -2:
        return 1
    else:
        return 0

df['u_c'] = df['sg_epb'] - df['sg_epb'].shift(1)
df['l_c'] = df['sg_epb'] - df['sg_epb'].shift(-1)

df['sg_smooth'] = df.apply(lambda x: smooth_savgol(x['sg_epb'], 
        x['u_c'], x['l_c']), axis=1)


def std_dev_check(df):

    #std dev filter to remove false positives
    #if stddev is greater than value, then use sg-filter
    #else overite sg-filter with 0
    
    window = 20
    df['Ne_pc'] = df['Ne'].pct_change(periods=1)
    df['Ne_stddev'] = df['Ne_pc'].rolling(window).std()
    df = df.dropna()

    value = 0.1
    if df['Ne_stddev'].max() > value:
        #print(f'greater than {value}')
        pass
    else:
        #print(f'smaller than {value}')
        df = df.loc[df['sg_smooth'] == 0]

        pass
        
    return df

#df = std_dev_check(df)

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
        x['sg_smooth']), axis=1)

#print(df)

scores = df.groupby('func_score').size()
precision = scores.iloc[3] / (scores.iloc[3] + scores.iloc[1])
recall = scores.iloc[3] / (scores.iloc[3] + scores.iloc[0])
f1 = 2*((precision*recall)/(precision+recall))
precision = "{:.2f}".format(precision) 
recall = "{:.2f}".format(recall) 
f1 = "{:.2f}".format(f1)

print('Test', test_id)
print('Scores',scores)
print ('Precision:', precision)
print ('Recall:', recall)
print('F1:',f1)

def plotNoStdDev(df):
        
        figs, axs = plt.subplots(ncols=1, nrows=5, figsize=(7,5), 
        dpi=90, sharex=True) #3.5 for single, #5.5 for double
        axs = axs.flatten()
        
        df = df[df['lat'].between(-10,30)]
        
        x = 'lat'
        hue = 's_id'

        ax0y = 'Ne'
        sns.lineplot(ax = axs[0], data = df, x = x, y =ax0y, 
                palette = 'bone',hue = hue, legend=False)

        ax1y = 'pot'
        sns.lineplot(ax = axs[0], data = df, x =x, y =ax1y,
                palette = 'Set1', hue = hue, legend=False)

        ax2y = 'pot_savgol'
        sns.lineplot(ax = axs[1], data = df, x = x, y =ax2y, 
                palette = 'bone', hue = hue, legend = False)

        ax3y = 'pot_resid'
        sns.lineplot(ax = axs[2], data = df, x = x, y =ax3y, 
                palette = 'Set2', hue = hue, legend = False)

        ax4y = 'epb_gt'
        sns.lineplot(ax = axs[3], data = df, x = x, y =ax4y, 
                palette = 'rocket', hue = hue, legend = False)

        ax5y = 'sg_smooth'
        sns.lineplot(ax = axs[4], data = df, x = x, y =ax5y, 
                palette = 'rocket', hue = hue, legend = False)

        #ax6y = 'sovgol_epb'
        #sns.lineplot(ax = axs[5], data = df, x = x, y =ax5y, 
        #        palette = 'rocket', hue = hue, legend = False)

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
        
        title = 'EPB Classifier testing. Sample:'

        axs[0].set_title(f'{title} {date_s} at {utc_s} to {date_e} at {utc_e}'
               #f'\n Precision: {precision}, Recall: {recall}, F1: {f1}' 
                ,fontsize = 11)

        den = r'cm$^{-3}$'
        axs[0].set_ylabel(f'{ax0y}')
        axs[0].tick_params(bottom = False)
        axs[0].set_yscale('log')
        #axs[0].set_ylim([1e6, 5e6])
        
        axs[1].set_ylabel(f'{ax2y}')
        axs[1].tick_params(bottom = False)
        #axs[2].set_yscale('log')
        
        axs[2].set_ylabel(f'{ax3y}')
        axs[2].tick_params(bottom = False)

        axs[3].set_ylabel(f'{ax4y}')
        axs[3].tick_params(bottom = False)

        axs[4].set_ylabel(f'{ax5y}')
        axs[4].tick_params(bottom = False)

        #axs[5].set_ylabel(f'{ax6y}')
        #axs[5].tick_params(bottom = False)

        #axs3.set_ylabel('Actual (Green)')

        ax = plt.gca()
        #ax.invert_xaxis()


        plt.tight_layout()
        plt.show()

plotNoStdDev(df)