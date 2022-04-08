'''
    The file applies the Savitzky-Golay filter to selected passes

    It counts the number of EPB events per day and drops dates which do
    not meet the criteria

    It outputs a new classified csv, which is an input for the ML pipeline

    Created by Sachin A. Reddy

    February 2022.
'''

import pandas as pd
from pathlib import Path
from datetime import date
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import seaborn as sns
pd.set_option('display.max_rows',  10) #or 10 or None

path = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/'
        'Missions/SWARM/Non-Flight Data/Analysis/Mar-22/data/solar_max/')

#path_ts = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/'
#        'Missions/SWARM/Non-Flight Data/Analysis/Mar-22/data/two_sat/')

dir_suffix = '2014'
load_all = str(path) + '/' + dir_suffix +'-data-2022-03-03.csv'
#load_all = str(path_ts) + '/' + 'AC-2014_2022-03-14.csv'

epb_mssl_output = str(path) +'/overfitting_test/'+'EPB-count-MSSL_'+dir_suffix+'.csv'
epb_ibi_output = str(path) +'/overfitting_test/'+'EPB-count-IBI_'+dir_suffix+'.csv'
#classified_output = str(path) +'/classified/'+'EPB-sg-classified_'+dir_suffix+'.csv'
#classified_output = str(path) +'/overfitting_test/'+'SG-applied_'+dir_suffix+'.csv'
#filter_classified_output = str(path) +'/classified/'+'EPB-sg-classified_stddev_filter_'+dir_suffix+'.csv'
filter_classified_output = str(path) +'/overfitting_test/'+'SG-filtered_'+dir_suffix+'.csv'


#for testing. Quicker to load
classified_output = str(path) +'/classified/'+'EPB-sg-classified_indie_'+dir_suffix+'.csv'

def open_all(filename):
    print('Loading data...')
    df = pd.read_csv(filename)
    return df

'''
df = open_all(load_all)
df = df[df['date'] == '2014-10-02']
df = df[['utc','mlt','lat','long','mlt','p_num','s_id','Ne']]
df = df.sort_values(by=['utc','s_id'], ascending=[True, True])
#df = df.groupby(['p_num']).mean()
df = df[df['utc'].between('00:20:00','00:40:00')]
print(df)

sns.lineplot(data=df, x='lat',y='Ne',hue='s_id')

plt.yscale('log')
plt.tight_layout()
plt.show()'''

class classify_epb():

    def ibi_mssl_epb_compare(self):
        '''
        This function applies the savitzky-golay to the swarm dataset
        It does so a pass at a time'''

        #for testing
        #df = open_all(classified_output)
        #df = df.drop(columns=['Ne_savgol','Ne_resid','sg_epb','sg_smooth'])
        #df = df.drop(columns=['pot_std','Te_std','Ti_std','alt'])
        #df = df.drop(columns=['Ne_pc','Ne_stddev'])
        #print(df)
        
        #full dataset
        df = open_all(load_all)

        print('Filtering data...')
        #df = df.drop(columns=['pot_std','Te_std','Ti_std','alt'])
        df = df[df['b_ind']!=-1]
        df = df[df['lat'].between(-35,35)]
        df = df[df['date'] == '2014-02-01']
        #df = df[df['p_num'] == 641]

        #print(df)
        
        df_sg = df.groupby('p_num')
        df_arr_count = []
        df_arr_sg = []

        for name, class_group in df_sg:
            
            try:

                print('Classifying pass', name)

                def classify_ne(df):

                    def savitzky_golay(df):
                        from scipy.signal import savgol_filter
                        df['Ne_savgol'] = savgol_filter(df['Ne'], window_length=23,
                            polyorder = 2) 
                        df['Ne_resid'] = df['Ne'] - df['Ne_savgol']
                        df.dropna()

                        return df

                    df = savitzky_golay(df)

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

                    def std_dev_check(df):

                       #std dev filter to remove false positives
                       #if stddev is greater than value, then use sg-filter
                       #else overite sg-filter with 0
                        
                        window = 20
                        df['Ne_pc'] = df['Ne'].pct_change(periods=1)
                        df['Ne_stddev'] = df['Ne_pc'].rolling(window).std()
                        df = df.dropna()

                        value = 0.2
                        if df['Ne_stddev'].max() > value:
                            #print(f'greater than {value}')
                            pass
                        else:
                            #print(f'smaller than {value}')
                            df = df.loc[df['sg_smooth'] == 0]

                            pass
                            
                        return df
                    
                    df = std_dev_check(df)

                    #append to list
                    df_arr_sg.append(df)
                    return df_arr_sg
                
                classified_epb_sg = classify_ne(class_group)
                

                #ibi_epb_count = mssl_epb_counter(classified_epb_sg, 'b_ind')

            except:
                print('CLASSIFICATION ERROR', name)
                pass
        
        #return dataframes
        #Why you must not concat in a loop: 
        #https://stackoverflow.com/questions/13784192/creating-an
        # -empty-pandas-dataframe-then-filling-it

        classified_df = pd.concat(classified_epb_sg, ignore_index=True)

        #Count the number of EPB's per pass
        #Assumes one per pass and each is unique (this isn't true)
        df_count = classified_df.groupby('p_num')
        
        df_arr_mssl_count = []
        df_arr_ibi_count = []
        #df_arr_mssl_cont = []

        
        for name, count_group in df_count:

            try:

                print('Counting pass', name)

                def mssl_epb_counter(df, classifier, list_op):

                    df = df.groupby(['date','p_num'], 
                            as_index=False)[classifier].sum()
                
                    def count_epb(x):
                        if x > 1:
                            return 1
                        else:
                            return 0

                    #if classifier == 'sg_smooth':
                    #    df = df[~df[classifier].between(1,50)]
                    #else:
                    #    pass

                    df['epb_num'] = df[classifier].apply(count_epb)
                    df = df.rename(columns={df.columns[2]:'epb_size'})
                    list_op.append(df)
                
                    #df = pd.concat(df_arr_count)
                    #df = df.drop_duplicates(subset=['p_num'])
                    #df = df.reset_index().drop(columns=['index'])

                    return list_op
        
                mssl_epb_count = mssl_epb_counter(count_group, 'sg_smooth', 
                        df_arr_mssl_count)
                ibi_epb_count = mssl_epb_counter(count_group, 'b_ind', 
                        df_arr_ibi_count)

            except:
                print('COUNTING ERROR', name)
                pass

        #return count dataframes
        mssl_epb_df = pd.concat(mssl_epb_count, ignore_index=True)
        ibi_epb_df = pd.concat(ibi_epb_count, ignore_index=True)

        #mssl_epb_df = 1
        #ibi_epb_df = 1

        #df = df.sort_values(by=['date','p_num'], ascending=[True,True])
        
        #Export the dataframes
        print('Exporting dataframes...')
        classified_df.to_csv(classified_output, index=False, header = True)
        #mssl_epb_df.to_csv(epb_mssl_output, index=False, header = True)
        #ibi_epb_df.to_csv(epb_ibi_output, index=False, header = True)
        
        print('Dataframe exported.')
        return classified_df, mssl_epb_df, ibi_epb_df

    def filter_epb(self,mssl_cat, ibi_cat):
        
        df_mssl = open_all(epb_mssl_output)
        df_ibi = open_all(epb_ibi_output)

        #print(df_mssl)

        df_mssl = df_mssl.groupby(['date'], 
                as_index=False)[mssl_cat].sum()

        df_ibi = df_ibi.groupby(['date'], 
                as_index=False)[ibi_cat].sum()

        df_m = df_mssl.merge(df_ibi, on =['date'], how='outer')

        def remove_zero_days(x, y):
            if x == 0 and y >= 1: #IBI and MSSL must have an EPB per day
                return 0
            elif x >= 1 and y == 0:
                return 0
            elif x == 0 and y == 0: #Drop days without epb
                return 0
            #elif x < 5 or y < 5: #keep only days with 5+ epb events
            #    return 0
            else:
                return 1
        
        df_m['date_check'] = df_m.apply(lambda x: remove_zero_days(x['epb_num_x'],
                x['epb_num_y']), axis=1)
        df_m = df_m[df_m['date_check'] !=0 ]


        #split them back up
        df_mssl = df_m[['date',mssl_cat+'_x']]
        df_ibi = df_m[['date',ibi_cat+'_y']]


        def rename(df):
            df = df.rename(columns={df.columns[1]:mssl_cat})
            return df

        df_mssl = rename(df_mssl)
        df_ibi = rename(df_ibi)

        #How many dates are removed by filtering
        dates_remaining = len(df_m.index)
        #print('days Remaining:', dates_remaining)
        #print('pc remaining', (dates_remaining/112)*100)

        #print(df_mssl, df_ibi)
        return df_mssl, df_ibi

    def pivot_epb(self, df_mssl, df_ibi, cat):

        print(df_mssl)

        def split_datetime(df):
            temp_df = df["date"].str.split("-", n = 2, expand = True)
            df["year"] = temp_df [0]
            df["month"] = temp_df [1]
            df["day"] = temp_df [2]
            #df['month'] = df['month'].replace({'02':'Feb','03':'Mar','09':'Sep',
            #        '10':'Oct'})
            return df
        
        
        df_mssl = split_datetime(df_mssl)
        year = df_mssl['year'].iloc[0]
        df_ibi = split_datetime(df_ibi)

        
        
        #retain the dates
        retained_dates = df_mssl['date'].to_list()

        def pivot_table(df, values, index):
            pivot_data = df.pivot_table(values=values,index=index,
                    columns="month", aggfunc=np.sum, dropna = False)

            return pivot_data

        pv_mssl = pivot_table(df_mssl, cat, "day")
        pv_ibi = pivot_table(df_ibi, cat, "day")


        def diff_pivots(pv_mssl, pv_ibi):

            df = pv_mssl.values - pv_ibi.values
            df = pd.DataFrame(df)
            df.index = df.index + 1
            df = df.rename(columns={0:'Feb',1:'Mar',2:'Sep',3:'Oct'})
            #df = df[(df > -3) & (df < 3)]
            
            return df

        mssl_diff_ibi = diff_pivots(pv_mssl, pv_ibi)

        print(mssl_diff_ibi)
        #print(pv_mssl)
        #print(pv_ibi)

        return mssl_diff_ibi, pv_mssl, pv_ibi, retained_dates, year
    
    def rebuild_sg_df(self, df, dates):

        #print('Remaining dates...', dates)
        df = open_all(classified_output)
        df = df[df['date'].isin(dates)]

        print('Exporting dataframe...')
        df.to_csv(filter_classified_output, index=False, header = True)
        print('Filtered dataframe exported.')

classify = classify_epb()
#print(full_df_mssl_classified)
#full_df_mssl_classified, mssl_epb_count, ibi_epb_count= classify.ibi_mssl_epb_compare()
#print('MSSL count\n',mssl_epb_count)
#print('IBI count\n',ibi_epb_count)

#EPB counter (how many EPB events per day)
#df_mssl, df_ibi = classify.filter_epb('epb_num','epb_num')
#mssl_ibi, pv_mssl, pv_ibi, retained_dates, year = classify.pivot_epb(df_mssl, df_ibi,"epb_num")
#classify.rebuild_sg_df(mssl_ibi, retained_dates)

#EPB continuous data (how large is each point?)
#df_mssl, df_ibi = classify.filter_epb('epb_size','epb_size')
#mssl_ibi, pv_mssl, pv_ibi, retained_dates, year = classify.pivot_epb(df_mssl, df_ibi,"epb_size")


def panel_plot(dpi):

        #print(df)

        df = open_all(classified_output)
        df = df[df['p_num'] == 2845]
        '''
        #df_gb = df.groupby(['utc']).mean()
        df = df[df['utc'].between('00:22:00','00:38:00')]
        df = df.sort_values(by=['utc','s_id'], ascending=[True,True])

        df_long = df[['utc','lat','long','s_id','Ne']]
        df_long = df_long.groupby(['utc','lat','long','s_id'],as_index=False).mean()

        df_long['long_diff'] = df_long['long'] - df_long['long'].shift(1)
        df_long = df_long.iloc[1::2]
        df_long['long_diff_km'] = abs(df_long ['long_diff'] * 111)

        #df = df[df['p_num'] == 2838]
        #df = full_df_mssl_classified
        print(df_long)'''

        
        figs, axs = plt.subplots(ncols=1, nrows=4, figsize=(8,4.5), 
        dpi=dpi, sharex=True) #3.5 for single, #5.5 for double
        axs = axs.flatten()

        x = 'lat'
        hue = 's_id'

        #palette = sns.palplot(sns.dark_palette((260, 75, 60), input="husl"))

        ax0y = 'Ne'
        sns.lineplot(ax = axs[0], data = df, x = x, y =ax0y, 
                palette = 'cubehelix' ,hue = hue, legend=True)

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

        title = 'EPB Classification: SWARM A & C (4-10s sep) \n'

        #seperation article
        #https://earth.esa.int/eogateway/news/swarm-s-orbital-dance-counter-
        #rotating-and-closer-for-the-benefit-of-science
    
        axs[0].set_title(f'{title} on {date_s} ({p_num})' 
                #f'\n Precision: {precision}, Recall: {recall}, F1: {f1}' 
                ,fontsize = 11)

        den = r'cm$^{-3}$'
        axs[0].set_ylabel(f'{ax0y}')
        axs[0].tick_params(bottom = False)
        axs[0].set_yscale('log')
        axs[0].set_ylabel(f'Ne \n ({den})')
        #axs[0].set_ylim([1e4, 4e6])
        
        axs[1].set_ylabel(f'{ax1y}')
        axs[1].tick_params(bottom = False)
        axs[1].set_ylabel('IBI')

        axs[2].set_ylabel(f'{ax2y}')
        axs[2].tick_params(bottom = False)
        axs[2].set_ylabel('MSSL')
        #axs[2].set_ylabel('Long Diff \n(km)')

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
        ax.invert_xaxis()

        plt.tight_layout()

        #save_fig = str(fig_path)+'/'+ str(date_s) + '_' + str(p_num) + '.png'
        
        #plt.savefig(save_fig)
        #print('Figure saved')

        plt.show()

panel_plot(90)

def heatmap(df, pv_mssl, pv_ibi, year):

    import seaborn as sns 
    from matplotlib import pyplot as plt
    from matplotlib.colors import LogNorm

    #cmap = "YlGnBu" #individual days
    cmap = "PiYG" #differentiation

    #ax = plt.gca()
    #cbar_ax = ax1.add_axes([.92, .3, .02, .4])  # <-- Create a colorbar axes

    f,(ax1,ax2,axdum1, ax3, axdum2) = plt.subplots(1,5, 
            gridspec_kw={'width_ratios':[1,1,0.25, 1,0.25]}, figsize=(6,6.5))

    f.suptitle(f'Number of EPB events per day in {year}'
            #'\n without std dev filter'
           ,fontsize=11)
    plt.subplots_adjust(top=0.85)

    axdum1.axis('off')
    axdum2.axis('off')
    #ax1.get_shared_y_axes().join(ax2,ax3)
    
    #ax1 = f.add_subplot(gs(wspace=1))
    
    axcmp1 = f.add_axes([0.57, 0.1, .02, 0.8]) #long
    #axcmp1 = f.add_axes([0.33, 0.08, .02, 0.22]) 
    g1 = sns.heatmap(pv_mssl,cmap="YlGnBu",ax=ax1, cbar_ax=axcmp1, annot=True)
    g1.set_title('MSSL',fontsize = 10)
    g1.set_ylabel('')
    g1.set_xlabel('')
    #g1.add_subplot(gs[0:2,0:2])

    #ax2 = f.add_subplot(gs[2:3])
    g2 = sns.heatmap(pv_ibi,cmap="YlGnBu", ax=ax2, cbar=False, annot=True)
    g2.set_title('IBI',fontsize = 10)
    g2.set_ylabel('')
    g2.set_xlabel('')
    g2.set_yticks([])

    axcmp2 = f.add_axes([0.91, 0.1, .02, 0.8])  # <-- Create a colorbar axes
    #axcmp2 = f.add_axes([0.63, 0.1, .02, 0.2])  # <-- Create a colorbar axes
    g3 = sns.heatmap(df,cmap="PiYG",ax=ax3, cbar_ax=axcmp2, center=0, annot=True)
    g3.set_title('MSSL - IBI',fontsize = 10)
    g3.set_ylabel('')
    g3.set_xlabel('')
    g3.set_yticks([])

    for ax in [g1,g2,g3]:
        tl = ax.get_xticklabels()
        ax.set_xticklabels(tl, rotation=0)
        tly = ax.get_yticklabels()
        ax.set_yticklabels(tly, rotation=0)

    plt.tight_layout(pad=1)
    plt.show()


    '''
    plt.figure(figsize = (3,6.6))
    sns.heatmap(df, annot=True, linewidths=0.1,
                fmt=".0f", cmap=cmap, center=0)
    
    plt.title(f'Number of EPB events in {date}, \n IBI vs. MSSL', fontsize=10.5)
    #plt.title(f'Number of EPB events in {date}, \n (MSSL Classifier)', fontsize=10.5)
    plt.xlabel(' ')
    plt.ylabel(' ')

    #ax.figure.axes[-1].set_ylabel(f'{log} counts per sec', size=9.5)
    plt.yticks(rotation = 0)

    plt.tight_layout()
    plt.show()'''

#df_mssl = open_all(epb_mssl_output)
#df_ibi = open_all(epb_ibi_output)
#heatmap(mssl_ibi, pv_mssl, pv_ibi, year)

def determine_epb_intensity():

    import plotly.express as px

    df = open_all(filter_classified_output)

    #print(df)
    
    #reformating the columns for better plottability
    #df = df[df['date'] == '2015-03-03']
    df['lat'] = df['lat'].round(0)
    df['long'] = df['long'].round(0)
    #df['long'] = (df['long'] /10).round().astype(int) * 10
    #df['lat'] = (df['lat'] / 10).round().astype(int) * 10
    df['mlt'] = df['mlt'].round(1)
    df["utc"] = df['utc'].str.slice(stop =-3)
    df["mon"] = df['date'].str.slice(start=5, stop =-3)
    df["day"] = df['date'].str.slice(start=8)
    #df = df[df['mon'] == "03"]

    def pot_check(x):

        if x > -0.7:
            return 1
        else:
            return 0
    
    #df['pot_den'] = df['pot'].apply(pot_check)


    #df['Ne_stddev'].max() > value:
    df_scat = df.groupby(['lat','long'], as_index=False)['sg_smooth'].sum()
    #print(df_scat)

    df_epb = df.groupby(['date','mlt','lat','long'], as_index=False)['sg_smooth'].sum()
    print(df_epb)
    df_pot = df.groupby(['date','mlt','lat', 'long'], as_index=False)[['pot']].mean()
    df_te = df.groupby(['date','lat', 'long'], as_index=False)['Te'].mean()

    #df = df.groupby(['date','lat', 'long'], as_index=False)['Te'].mean()

    df = df_epb.merge(df_pot, on =['date','mlt','lat','long'], how='outer')
    df['corr'] =  df['sg_smooth'].rolling(window=10).corr(df['pot'])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    '''
    plt.figure(figsize=(14,5),dpi=90)
    pivot_df = df.pivot_table(values="corr",index="lat",
                    columns="long", aggfunc=np.mean, dropna = False)
    print(pivot_df)
    sns.heatmap(pivot_df, cmap="PiYG", center=0)
    plt.xticks(rotation=0) 
    plt.tight_layout()
    plt.show()'''

    def corr_check(x):
        if 0.75 < x < 1:
            return 1
        else:
            return 0

        '''
        elif -0.5 < x < 0:
            return 2
        elif 0 < x < 0.5:
            return 3
        elif -0.5 < 1:
            return 4
        else:
            return 0'''
        
        #print(-1 < num < 4)
        '''
        if x > 0:
            return 1
        else:
            return 0'''
    
    df['corr_den'] = df['corr'].apply(corr_check)

    '''orbyts/
    #fig = px.scatter_mapbox(df_scat, lat='lat', lon='long',
    fig = px.scatter_geo(df_scat, lat='lat', lon='long',
                    size="sg_smooth", # size of markers, "pop" is one of the columns of gapminder
                    center=dict(lat=0, lon=0),
                    color="sg_smooth",
                    #basemap_visible = False,
                    labels = {'sg_smooth':'EPB Flag count'},
                    #title= "Size of EPB"
                    #animation_frame = "p_num"
                    #color_continuous_scale=px.colors.sequential.Turbo,
                    #mapbox_style="carto-darkmatter"
                    )'''




    #window = 50
    #mssl_epb_intensity['corrs'] =  mssl_epb_intensity['sg_smooth'].corr(mssl_epb_intensity['pot'])

    #df['# of EPB flags'] = df['sg_smooth']
    
    #mssl_epb_intensity = df.groupby(['date','mlt','lat','long'], as_index=False)['# of EPB flags'].sum()
    #        as_index=False)['pot'].sum()

    #print(df)

    #print(mssl_epb_intensity['pot'].mean())
    #window = 50
    #mssl_epb_intensity['corrs'] =  mssl_epb_intensity['sg_smooth'].corr(mssl_epb_intensity['pot'])
    #mssl_epb_intensity['corrs'] = mssl_epb_intensity['sg_smooth'].rolling(window=window).corr(mssl_epb_intensity['pot'])
    #mssl_epb_intensity = mssl_epb_intensity.dropna()
    #print(mssl_epb_intensity)

    
    import plotly.express as px
    fig = px.density_mapbox(df_epb, lat='lat', lon='long', z='sg_smooth', radius=7,
                    center=dict(lat=0, lon=0), zoom=1.5, 
                    #range_color=[-1.5, 0.5], #pot (mean)
                    #range_color = [-3, 3], #pot (sum)
                    range_color=[0, 35], #number of MSSL epb
                    labels = {"sg_smooth":"Density of EPBs"},
                    #range_color=[1100, 3000], #Te
                    #range_color = [0.5,2],
                    #hover_name = 'date',
                    #animation_frame = 'date',
                    #color_continuous_scale=px.colors.sequential.Viridis,
                    #mapbox_style="carto-darkmatter"
                    mapbox_style ="carto-positron"
                    )

    
    #fig = px.scatter_geo(df_epb, lat='lat',
    #                size='sg_smooth',
    #                projection="natural earth")'''

    fig.update_layout(
            #legend_title="Legend Title",
            font=dict(size=20))

    fig.update_traces(visible=True)
    fig.show()
    #fig_2.show()
    
    return fig

#full_df_mssl_classified = open_all(classified_output)
#print(full_df_mssl_classified)
#determine_epb_intensity()

def train_test_class(dpi):

        #print(df)

        df = open_all(classified_output)
        
        p_num = 1689
        df = df[df['p_num'] == p_num]
        
        df_exp = df[['date','utc','mlt','lat','long','s_id','p_num','Ne','Ti','pot']]
        df_exp['id'] = 16
        df_exp['epb_gt'] = 0
        df_exp = df_exp.reset_index().drop(columns=['index'])

        date = df_exp['date'].iloc[0]
        path_e = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Non-Flight Data/Analysis/Mar-22/data/train-test-sg/new_additions/'
        
        export = path_e + date + '_' + str(p_num) + '_test_train.csv'
        #df_exp.to_csv(export, index=False, header = True)

        print(df_exp)

        

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
        #axs[0].set_ylim([1e4, 4e6])
        
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
        ax.set_xlim([-30, 30])
        #ax.invert_xaxis()

        plt.tight_layout()

        #save_fig = str(fig_path)+'/'+ str(date_s) + '_' + str(p_num) + '.png'
        
        #plt.savefig(save_fig)
        #print('Figure saved')

        plt.show()

#train_test_class(90)