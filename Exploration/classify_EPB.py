'''

    Created by Sachin A. Reddy

    February 2022.
'''


import pandas as pd
from pathlib import Path
from datetime import date
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


path = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/'
        'Missions/SWARM/Non-Flight Data/Analysis/Feb-22/data/solar_max/')


def open_all(filename):

    #filename = '2015-data-2022-02-14.csv'
    df = pd.read_csv(str(path) + '/' +filename)

    return df

def count_ibi_epb(df):
    #This functon counts the number of EPB's per day. It assumes there is only
    #1 EPB per pass and a new pass constitutes a new EPB. This might not always
    #be the case.

    df = df[df['b_ind']!=-1]
    df = df.groupby(['date','p_num'], as_index=False)[['b_prob','b_ind']].mean()

    def counter(x):
        if x > 1:
            return 1
        else:
            return 0

    df = df.sort_values(by=['date','p_num'], ascending=[True,True])
    df = df[df['b_prob']>0.4].reset_index().drop(columns=['index'])
    #df = df[df['b_ind']>300].reset_index().drop(columns=['index'])

    return df
    
def select_date(df, date, p_num):
    df = df[df['date']== date]
    df = df[df['p_num']== p_num ]

    output = str(path) + '/' + date + str(p_num) +'.csv'
    df.to_csv(output, index=False, header = True)

    return df

selection = 'mult'
target_date = "2015-02-08"
p_num = 3275

if selection == 'multi':

    filename = '2015-data-2022-02-14.csv'
    df = open_all(filename)

    #view EPBs
    epb_df = count_ibi_epb(df)
    #print(epb_df)

    #select data and export
    df = select_date(df, target_date, p_num)
    print(df)
    
else:
    filename = str(path) + '/' + target_date + str(p_num) + '.csv'
    df = pd.read_csv(filename)

    print(df)


def norm_data(df):
    from sklearn.preprocessing import StandardScaler
    x_data = df[['Ne','pot']]
    scaler = StandardScaler()
    scaler.fit(x_data) #compute mean for removal and std
    x_data = scaler.transform(x_data)
    ne_scale = [a[0] for a in x_data]
    df['Ne_scale'] = ne_scale
    return df

df = norm_data(df)
#print(df)

def savitzky_golay(df):
    from scipy.signal import savgol_filter
    df['Ne_savgol'] = savgol_filter(df['Ne_scale'], window_length=23,
        polyorder = 2) 
    df['Ne_resid'] = df['Ne_scale'] - df['Ne_savgol']
    df.dropna()

    def sovgol_epb(x):
        #if x > 10000 or x < -10000: non-norm
        if x > 0.05 or x < -0.05: #norm
            return 1
        else:
            return 0

    df['savgol_epb'] = df.apply(lambda x: sovgol_epb(x['Ne_resid']), axis=1)
    #df = df[['date','utc','lat','long','Ne','','s_id','Ne_savgol','Ne_resid',
    #        'b_ind','savgol_epb']]

    return df

df = savitzky_golay(df)
#print(df)

def plotSavGol(df):

        df = df[df['lat'].between(-30,30)]
        print(df)

        figs, axs = plt.subplots(ncols=1, nrows=5, figsize=(8,5), 
        dpi=90, sharex=True) #3.5 for single, #5.5 for double
        axs = axs.flatten()

        x = 'lat'
        hue = 's_id'

        ax0y = 'Ne'
        sns.lineplot(ax = axs[0], data = df, x = x, y =ax0y, 
                palette = 'bone',hue = hue, legend=False)

        ax1y = 'Ne_savgol'
        sns.lineplot(ax = axs[1], data = df, x =x, y =ax1y,
                palette = 'Set1', hue = hue, legend=False)
 
        ax2y = 'Ne_scale'
        sns.lineplot(ax = axs[1], data = df, x = x, y =ax2y, 
                palette = 'bone', hue = hue, legend = False)

        ax3y = 'Ne_resid'
        sns.lineplot(ax = axs[2], data = df, x = x, y =ax3y, 
                palette = 'Set2', hue = hue, legend = False)

        ax4y = 'b_prob'
        sns.lineplot(ax = axs[3], data = df, x = x, y =ax4y, 
                palette = 'rocket', hue = hue, legend = False)

        ax4y = 'savgol_epb'
        sns.lineplot(ax = axs[4], data = df, x = x, y =ax4y, 
                palette = 'rocket', hue = hue, legend = False)

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

plotSavGol(df)

def test():
    #df = df[df['p_num'] == 1]
    #df = df[df['b_ind'] == 1]
    #df = df.describe()

    #df = df[df['date']== '2015-02-01']
    df = df[df['b_ind']!= -1]
    #df = df[df['b_ind'] == 0]
    #df = df[~df['mlt'].between(6,18)]

    #epb_count = df.groupby(['date','p_num'])['b_ind'].count()
    df = df.groupby(['date','p_num'], as_index=False)['b_ind'].sum()
    
    def count_epb(x):
        if x > 1:
            return 1
        else:
            return 0

    temp_df = df["date"].str.split("-", n = 2, expand = True)
    df["year"] = temp_df [0]
    df["month"] = temp_df [1]
    df["day"] = temp_df [2]
    #df = df[::60]
    df = df.reset_index().drop(columns=['index'])
    df = df.sort_values(by=['date','p_num'], ascending=[True,True])

    df['epb'] = df['b_ind'].apply(count_epb)
    df = df[df['b_ind']>300]

    print(df)    
    
    '''
    pivot_data = df.pivot_table(values="epb",index="day",columns="month", 
            aggfunc=np.sum, dropna = False)
    print(pivot_data)

    import seaborn as sns 
    from matplotlib import pyplot as plt
    from matplotlib.colors import LogNorm
    
    plt.figure(figsize = (3,6.6))
    sns.heatmap(pivot_data, annot=True, linewidths=0.1, vmin=0,
             fmt=".0f", cmap="YlGnBu")

    plt.title('Number of EPB events in 2014 \n (IBI Classifier)', 
            fontsize=10.5)
    plt.xlabel('Month')
    plt.ylabel('Day')
    plt.yticks(rotation = 0)

    plt.tight_layout()
    plt.show()'''

#1 Classifer 
#Normalise the data
# Savitzy-Golar Filter

#2 Classifer
# Std dev on Ne

# Correlation between them

#Plot panels
# EPB IBI
# EPB MSSL
# Ne