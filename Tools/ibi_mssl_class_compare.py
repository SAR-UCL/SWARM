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

dir_suffix = '2015'

load_all = str(path) + '/' + '2015-data-2022-02-16.csv'
epb_output = str(path) +'/EPB_counts/'+'EPB-count-MSSL_'+dir_suffix+'.csv'
classified_output = str(path) +'/classified/'+'EPB-sg-classified_'+dir_suffix+'.csv'

def open_all(filename):
    print('Loading data...')
    df = pd.read_csv(filename)
    return df

def ibi_mssl_epb_compare():

    df = open_all(load_all)
    print('Filtering data...')
    df = df[df['b_ind']!=-1]
    df = df[df['lat'].between(-35,35)]
    #df = df[df['lat'].between(-30,30)]
    #df = df[df['date'] == '2015-02-14']
    
    df_new = df.groupby('p_num')

    df_arr_count = []
    df_arr_sg = []

    for name, num_group in df_new:

        try:

            print('Classifying pass', name)

            def savitzky_golay(df):
                from scipy.signal import savgol_filter
                df['Ne_savgol'] = savgol_filter(df['Ne'], window_length=23,
                    polyorder = 2) 
                df['Ne_resid'] = df['Ne'] - df['Ne_savgol']
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

                df_arr_sg.append(df)
                df = pd.concat(df_arr_sg)
                df = df.reset_index().drop(columns=['index'])

                return df

            classified_epb_sg = savitzky_golay(num_group)
      
            def mssl_epb_counter(df):

                df = df.groupby(['date','p_num'], 
                        as_index=False)['sg_smooth'].sum()
            
                def count_epb(x):
                    if x > 1:
                        return 1
                    else:
                        return 0

                df = df[~df['sg_smooth'].between(1,50)]
                df['epb_num'] = df['sg_smooth'].apply(count_epb)

                df_arr_count.append(df)
                df = pd.concat(df_arr_count)

                df = df.drop_duplicates(subset=['p_num'])
                df = df.reset_index().drop(columns=['index'])

                return df
            
            mssl_epb_count = mssl_epb_counter(classified_epb_sg)

        except:
            print('ERROR', name)
            pass
    
    #df = df.sort_values(by=['date','p_num'], ascending=[True,True])
    classified_epb_sg.to_csv(classified_output, index=False, header = True)
    mssl_epb_count.to_csv(epb_output, index=False, header = True)
    #print('EPB Exported.')
    return classified_epb_sg, mssl_epb_count

full_df_mssl_classified, mssl_epb_count = ibi_mssl_epb_compare()
#print(full_df_mssl_classified)
#print(mssl_epb_count)

#Load Data for Heatmap
#df = open_all(EPB_output)
#df = df[df['p_num'] == 2313]
#df = df.loc[((df['p_num'] == 2301 ) | (df['p_num'] == 2313))]
#df = df.fillna(0)
#print(df)

def panel_plot(df, dpi):

        #print(df)

        figs, axs = plt.subplots(ncols=1, nrows=4, figsize=(8,4.5), 
        dpi=dpi, sharex=True) #3.5 for single, #5.5 for double
        axs = axs.flatten()

        x = 'lat'
        hue = 'p_num'

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

        p_num = df['p_num'].iloc[0]
        date_s = df['date'].iloc[0]
        lat_s = df['lat'].iloc[0]
        lat_e = df['lat'].iloc[-1]


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

        #save_fig = str(fig_path)+'/'+ str(date_s) + '_' + str(p_num) + '.png'
        
        #plt.savefig(save_fig)
        #print('Figure saved')

        plt.show()

#panel_plot(df, 90)

def heatmap():

    df = open_all(epb_output)


    temp_df = df["date"].str.split("-", n = 2, expand = True)
    df["year"] = temp_df [0]
    df["month"] = temp_df [1]
    df["day"] = temp_df [2]
    #df.fillna(0)

    pivot_data = df.pivot_table(values="epb_num",index="day",columns="month", 
            aggfunc=np.sum, dropna = False)

    print(pivot_data)

    import seaborn as sns 
    from matplotlib import pyplot as plt
    from matplotlib.colors import LogNorm

    plt.figure(figsize = (3,6.6))
    sns.heatmap(pivot_data, annot=True, linewidths=0.1, vmin=0,
                fmt=".0f", cmap="YlGnBu")

    plt.title('Number of EPB events in 2015 \n (MSSL Classifier)', 
            fontsize=10.5)
    plt.xlabel('Month')
    plt.ylabel('Day')
    plt.yticks(rotation = 0)

    plt.tight_layout()
    plt.show()

heatmap()