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
epb_ibi = str(path) +'/EPB_counts/'+'EPB-count-IBI_'+dir_suffix+'.csv'
classified_output = str(path) +'/classified/'+'EPB-sg-classified_'+dir_suffix+'.csv'

def open_all(filename):
    print('Loading data...')
    df = pd.read_csv(filename)
    return df

def ibi_mssl_epb_compare():

    #for testing
    df = open_all(classified_output)
    df = df.drop(columns=['Ne_savgol','Ne_resid','sg_epb','sg_smooth'])

    print(df)

    #df = open_all(load_all)

    print('Filtering data...')
    df = df[df['b_ind']!=-1]
    df = df[df['lat'].between(-35,35)]
    #df = df[df['lat'].between(-30,30)]
    #df = df[df['date'] == '2015-02-14']
    #df = df[df['p_num'] == 3474]
    
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

                #df = df.drop(columns=['u_c','l_c','pot_std','Te_std','Ti_std'])
                df = df.drop(columns=['u_c','l_c'])

                df_arr_sg.append(df)

                return df_arr_sg
            
            classified_epb_sg = savitzky_golay(num_group)
            
            #ibi_epb_count = mssl_epb_counter(classified_epb_sg, 'b_ind')

        except:
            print('ERROR', name)
            pass
    

    #return dataframes
    classified_df = pd.concat(classified_epb_sg, ignore_index=True)

    df_count = classified_df.groupby('p_num')
    
    df_arr_mssl_count = []
    df_arr_ibi_count = []

    for name, num_group in df_count:

            #df_sg = df.groupby(['date','p_num'], as_index=False)['sg_smooth'].sum()

            #df_sg = df_sg[~df_sg['sg_smooth'].between(1,50)]
            #df_sg['epb_num'] = df_sg['sg_smooth'].apply(count_epb)


            def mssl_epb_counter(df, classifier, list_op):

                df = df.groupby(['date','p_num'], 
                        as_index=False)[classifier].sum()
            
                def count_epb(x):
                    if x > 1:
                        return 1
                    else:
                        return 0

                df = df[~df[classifier].between(1,50)]
                df['epb_num'] = df[classifier].apply(count_epb)

                '''
                #weirdly can't turn string into argument. Doesn't like it
                if classifer == 'sg_smooth':
                    df = df[~df['sg_smooth'].between(1,50)]
                    df['epb_num'] = df['sg_smooth'].apply(count_epb)
                else:
                    pass
                    df = df[~df['b_ind'].between(1,50)]
                    df['epb_num'] = df['b_ind'].apply(count_epb)'''

                list_op.append(df)

                #df = pd.concat(df_arr_count)
                #df = df.drop_duplicates(subset=['p_num'])
                #df = df.reset_index().drop(columns=['index'])

                return list_op
     
            mssl_epb_count = mssl_epb_counter(num_group, 'sg_smooth', 
                    df_arr_mssl_count)
            ibi_epb_count = mssl_epb_counter(num_group, 'b_ind', 
                    df_arr_ibi_count)

    mssl_epb_df = pd.concat(mssl_epb_count, ignore_index=True)
    ibi_epb_df = pd.concat(mssl_epb_count, ignore_index=True)

    #classified_df = classified_epb_sg

    #df = df.sort_values(by=['date','p_num'], ascending=[True,True])
    
    #Export the dataframes
    #classified_df.to_csv(classified_output, index=False, header = True)
    #mssl_epb_count.to_csv(epb_output, index=False, header = True)
    
    print('EPB Exported.')
    return classified_df, mssl_epb_df

print(ibi_mssl_epb_compare())

#full_df_mssl_classified, mssl_epb_count, ibi_epb_count = ibi_mssl_epb_compare()
#print(mssl_epb_count)
#print(ibi_epb_count)
#print(full_df_mssl_classified)
#print(mssl_epb_count)

#Load Data for Heatmap
#df = open_all(EPB_output)
#df = df[df['p_num'] == 2313]
#df = df.loc[((df['p_num'] == 2301 ) | (df['p_num'] == 2313))]
#df = df.fillna(0)
#print(df)

def panel_plot(dpi):

        #print(df)

        df = open_all(classified_output)
        #df = full_df_mssl_classified
        
        #df = df[df['p_num'] == 2925]

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
    
        axs[0].set_title(f'{title} on {date_s} ({p_num})' 
                #f'\n Precision: {precision}, Recall: {recall}, F1: {f1}' 
                ,fontsize = 11)

        den = r'cm$^{-3}$'
        axs[0].set_ylabel(f'{ax0y}')
        axs[0].tick_params(bottom = False)
        axs[0].set_yscale('log')
        axs[0].set_ylabel(f'Ne \n ({den})')
        axs[0].set_ylim([1e4, 1e6])
        
        axs[1].set_ylabel(f'{ax1y}')
        axs[1].tick_params(bottom = False)
        axs[1].set_ylabel('IBI')

        axs[2].set_ylabel(f'{ax2y}')
        axs[2].tick_params(bottom = False)
        axs[2].set_ylabel('MSSL')

        axs[3].set_ylabel(f'{ax3y}')
        #axs[3].tick_params(bottom = False)
        axs[3].tick_params(left = False)

        axs[3].set_ylabel('UTC')
        axs4.set_ylabel('MLT')

        n = len(df) // 3
        [l.set_visible(False) for (i,l) in 
                enumerate(axs[3].yaxis.get_ticklabels()) if i % n != 0]

        ax = plt.gca()
        ax.set_xlim([-40, 40])
        #ax.invert_xaxis()

        plt.tight_layout()

        #save_fig = str(fig_path)+'/'+ str(date_s) + '_' + str(p_num) + '.png'
        
        #plt.savefig(save_fig)
        #print('Figure saved')

        plt.show()

#panel_plot(90)

def heatmap():

    #load data
    df_mssl = open_all(epb_output)
    df_ibi = open_all(epb_ibi)

    def split_datetime(df):
        temp_df = df["date"].str.split("-", n = 2, expand = True)
        #df["year"] = temp_df [0]
        df["month"] = temp_df [1]
        df["day"] = temp_df [2]

        return df
    
    df_mssl = split_datetime(df_mssl)
    #df_ibi = split_datetime(df_ibi)

    def pivot_table(df):
        pivot_data = df.pivot_table(values="epb_num",index="day",
                columns="month", aggfunc=np.sum, dropna = False)

        return pivot_data

    pv_mssl = pivot_table(df_mssl)
    pv_ibi = pivot_table(df_ibi)

    def diff_pivots(df):

        delta = pv_mssl.values - pv_ibi.values
        delta = pd.DataFrame(delta)
        delta.index = delta.index + 1
        delta = delta.rename(columns={0:'Feb',1:'Mar',2:'Sep',3:'Oct'})
    #print(delta)
    
    import seaborn as sns 
    from matplotlib import pyplot as plt
    from matplotlib.colors import LogNorm

    plt.figure(figsize = (3,6.6))
    sns.heatmap(delta, annot=True, linewidths=0.1,
                fmt=".0f", cmap="PiYG", center=0)

    plt.title('Number of EPB events in 2015 \n (MSSL Classifier)', 
            fontsize=10.5)
    plt.xlabel(' ')
    plt.ylabel(' ')
    plt.yticks(rotation = 0)

    plt.tight_layout()
    plt.show()

#heatmap()