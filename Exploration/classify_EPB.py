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
pd.set_option('display.max_rows', 10) #or 10 or None


path = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/'
        'Missions/SWARM/Non-Flight Data/Analysis/Feb-22/data/solar_max/')

def open_all(filename):
    df = pd.read_csv(str(path) + '/' +filename)
    return df

filename = '2015-data-2022-02-15.csv'
#df = open_all(filename)
#df = df[df['b_ind']!=-1]
#print(df)

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

    output = str(path)+ '/by_date/' + date + str(p_num) +'.csv'
    df.to_csv(output, index=False, header = True)

    return df


selection = 'multi'
target_date = "2015-02-14"
p_num = 2322

if selection == 'mult':

    filename = '2015-data-2022-02-15.csv'
    df = open_all(filename)

    #view EPBs
    epb_df = count_ibi_epb(df)
    #print(epb_df)

    #select data and export
    df = select_date(df, target_date, p_num)
    df = df[df['b_ind']!=-1]
    #print(df)
    
else:
    filename = str(path) + '/by_date/' + target_date + str(p_num) + '.csv'
    df = pd.read_csv(filename)
    df = df[df['b_ind']!=-1]

    #print(df)

def norm_data(df):
    from sklearn.preprocessing import StandardScaler
    x_data = df[['Ne','pot']]
    scaler = StandardScaler()
    scaler.fit(x_data) #compute mean for removal and std
    x_data = scaler.transform(x_data)
    ne_scale = [a[0] for a in x_data]
    df['Ne_scale'] = ne_scale
    return df

#df = norm_data(df)
#print(df)

df['Ne_scale'] = df['Ne']

def savitzky_golay(df):
    from scipy.signal import savgol_filter
    df['Ne_savgol'] = savgol_filter(df['Ne_scale'], window_length=23,
        polyorder = 2) 
    df['Ne_resid'] = df['Ne_scale'] - df['Ne_savgol']
    df.dropna()

    def sovgol_epb(x):
        if x > 10000 or x < -10000: #non-norm
        #if x > 0.05 or x < -0.05: #norm
            return 1
        else:
            return 0

    df['sg_epb'] = df.apply(lambda x: sovgol_epb(x['Ne_resid']), axis=1)

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

    df = df.drop(columns=['u_c','l_c','pot_std','Te_std','Ti_std'])

    return df

df = savitzky_golay(df)

#print(df)

def performance_check(df):

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

    df['func_score'] = df.apply(lambda x: check_function(x['b_ind'], 
            x['sg_smooth']), axis=1)

    scores = df.groupby('func_score').size()

    precision = scores.iloc[3] / (scores.iloc[3] + scores.iloc[1])
    recall = scores.iloc[3] / (scores.iloc[3] + scores.iloc[0])
    f1 = 2*((precision*recall)/(precision+recall))
    precision = "{:.2f}".format(precision) 
    recall = "{:.2f}".format(recall) 
    f1 = "{:.2f}".format(f1)

    #print('Scores',scores)
    print ('Precision:', precision)
    print ('Recall:', recall)
    print('F1:',f1)

    return precision, recall, f1

#precision, recall, f1 = performance_check(df)
#print(performance)

def plotSavGol(df):

        #print(df)

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

        ax3y = 'b_ind'
        sns.lineplot(ax = axs[2], data = df, x = x, y =ax3y, 
                palette = 'Set2', hue = hue, legend = False)

        ax4y = 'sg_epb'
        sns.lineplot(ax = axs[3], data = df, x = x, y =ax4y, 
                palette = 'rocket', hue = hue, legend = False)

        ax4y = 'sg_smooth'
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

#plotSavGol(df)

def plotSavGol_formal(df, fig_path):

        #print(df)

        figs, axs = plt.subplots(ncols=1, nrows=4, figsize=(8,4.5), 
        dpi=300, sharex=True) #3.5 for single, #5.5 for double
        axs = axs.flatten()

        x = 'lat'
        hue = 's_id'

        #palette = sns.palplot(sns.dark_palette((260, 75, 60), input="husl"))

        ax0y = 'Ne'
        sns.lineplot(ax = axs[0], data = df, x = x, y =ax0y, 
                palette = 'cubehelix' ,hue = hue, legend=False)

        ax1y = 'b_ind'
        sns.lineplot(ax = axs[1], data = df, x =x, y =ax1y,
                palette = 'cubehelix', hue = hue, legend=False)
 
        ax2y = 'sg_smooth'
        sns.lineplot(ax = axs[2], data = df, x = x, y =ax2y, 
                palette = 'cubehelix', hue = hue, legend = False)

        ax3y = 'utc'
        sns.lineplot(ax = axs[3], data = df, x = x, y =ax3y, 
                palette = 'cubehelix', hue = hue, legend = False)

        ax4y = 'mlt'
        axs4 = axs[3].twinx()
        sns.lineplot(ax = axs4, data = df, x = x, y =ax4y, 
                palette = 'dark', hue = hue, legend = False)

        axs4.tick_params(axis = 'y', colors='darkblue', which='both')  
        axs4.yaxis.label.set_color('darkblue')

        #axs[3].yaxis.label.set_color('darkgoldenrod')
        #axs[3].tick_params(axis = 'y', colors='darkgoldenrod', which='both')  

        date_s = df['date'].iloc[0]
        date_e = df['date'].iloc[-1]
        utc_s = df['utc'].iloc[0]
        utc_e = df['utc'].iloc[-1]
      
        lat_s = df['lat'].iloc[0]
        lat_e = df['lat'].iloc[-1]

        epb_len = (lat_s - lat_e) * 110
        epb_len = "{:.0f}".format(epb_len)

        title = 'EPB Classification: IBI Processor vs. MSSL'
        
        #title = 'EPB Classifier testing. Sample:'

        axs[0].set_title(f'{title} on {date_s} ({p_num})' 
                #f'\n Precision: {precision}, Recall: {recall}, F1: {f1}' 
                ,fontsize = 11)

        den = r'cm$^{-3}$'
        axs[0].set_ylabel(f'{ax0y}')
        axs[0].tick_params(bottom = False)
        axs[0].set_yscale('log')
        axs[0].set_ylabel(f'Density \n ({den})')
        
        axs[1].set_ylabel(f'{ax1y}')
        axs[1].tick_params(bottom = False)
        axs[1].set_ylabel('IBI \n Method')

        axs[2].set_ylabel(f'{ax2y}')
        axs[2].tick_params(bottom = False)
        axs[2].set_ylabel('MSSL \n Method')

        axs[3].set_ylabel(f'{ax3y}')
        #axs[3].tick_params(bottom = False)
        axs[3].tick_params(left = False)

        axs[3].set_ylabel('UTC')
        axs4.set_ylabel('MLT')

        n = len(df) // 3
        [l.set_visible(False) for (i,l) in 
                enumerate(axs[3].yaxis.get_ticklabels()) if i % n != 0]

        ax = plt.gca()
        #ax.invert_xaxis()

        plt.tight_layout()

        save_fig = str(fig_path)+'/'+ str(date_s) + '_' + str(p_num) + '.png'
        plt.savefig(save_fig)
        print('Figure saved')

        #plt.show()


fig_path = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/'
    r'Missions/SWARM/Non-Flight Data/Analysis/Feb-22/plots/solar_max/valentines/') 

plotSavGol_formal(df, fig_path)

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