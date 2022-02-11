
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
pd.set_option('display.max_rows', 10) #or 10 or None
import seaborn as sns
from datetime import date

#Load exported .hdf files
#hdf_path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Non-Flight Data/Analysis/Jan-22/data/April-16/'
path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Non-Flight Data/Analysis/Feb-22/data/april-mag/'
#hdf_path = r'/Users/sr2
# 
# 
# /OneDrive - University College London/PhD/Research/Missions/SWARM/Non-Flight Data/Analysis/Jan-22/data/decadal/'

# file_name = 'joined-data-2022-01-13.h5'
# pathfile = hdf_path+file_name
# df = pd.read_hdf(pathfile)
# #df = df[df['lat'] == 34.448736]
# df = df[df['date'] == '2016-04-04']
# df = df[df['s_id'] == 'A']
# #df = df[:2700:] #45 min
# #df = df[:5400:] #90min

def pass_count(df):
    ml = []
    start = 0
    for i in range(len(df.index)):
        if i % 2700 == 0:
            start +=1
        else:
            pass
        ml.append(start)
    return ml

#counter = pass_count(df)
#df['pass'] = counter
#print(df)


class WrangleData():
    
    def __init__(self, pathfile, select_date=None):
        df = pd.read_csv(pathfile)
        self.df = df if select_date is None else df[df['date'] == select_date]

    @classmethod  # method that can be called before obj is created
    def frompath(cls, path, select_date=None):
        today =  str(date.today())
        #file_name = 'joined-data_2022-01-13'+ today +'.h5'
        file_name = 'April-16-data-'+today+'.csv'
        return cls(path+file_name, select_date)


    def transform_EPB(self, sat, s_time, e_time, s_lat, e_lat):

        #Filter the data
        #self.df = self.df[self.df['b_ind']!= -1] #remove non-useful data
        #self.df = self.df[self.df['long'].between(10,180)] #remove the SSA
        #self.df = self.df[~self.df['mlt'].between(6,18)] #Nightime only
        self.df = self.df[self.df['s_id'] == sat]
        self.df = self.df[self.df['utc'].between(s_time, e_time)]
        self.df = self.df[self.df['lat'].between(s_lat,e_lat)] #EPB region 

        #self.df['b_sum'] = np.sqrt(self.df['nec_x']**2 + self.df['nec_y']**2 + self.df['nec_z']**2)
        self.df['b_mu'] = self.df['f'] / (np.sqrt(self.df['nec_x']**2 + self.df['nec_y']**2 + self.df['nec_z']**2))
        
        def new_classifier(df):

                def classify_EPB(ne_std, b_ind, ti_std, pot_std):
                        if ne_std > 0.01 and ti_std > 0.008 and pot_std > 0.01:
                                return 1
                        else:
                                return 0

                #New EPB classification
                df['epb'] = df.apply(lambda x: classify_EPB(x.Ne_std, 
                        x.b_ind, x.Ti_std, x.pot_std), axis=1)
                #df = df[df['epb'] == 1]

                #Calculate EPB start, middle and end
                epb_cat = 'epb'
                df['change'] = df[epb_cat] - df[epb_cat].shift(1)
                df = df.replace({'change':{-1:2}})
                df['temp'] = np.where((df[epb_cat] == 1) 
                        & (df['change'] == 0), 1.5, 0)

                #Reassign s, m & e to one column
                #https://stackoverflow.com/questions/39109045/
                #numpy-where-with-multiple-conditions
                conditions = [df['change'] == 1, df['temp'] == 1.5, 
                        df['change'] ==2]
                output = [1,1.5,2]
                df['cat'] = np.select(conditions, output, default = '0')

                # if epb_only == True:
                #     df = df[df['cat'].between('1','2')]
                # else:
                #     pass

                df = df.reset_index().drop(columns=['change',
                        'temp','index'], axis=1)


                return df
        
        #df = new_classifier(self.df)
        #return df

 
        #Determine range
        def pre_post_EPB(df):#
                #cols = df.columns
                cols = df.loc[:, df.columns != 'b_ind']
                cols = df.loc[:, df.columns != 'b_prob']
                cols = df.loc[:, df.columns != 'epb']
                #print(cols)
                #cols = cols.columns
                cols = ['cat']
                df_pre = df.loc['1':,cols].head(25)

                print('df_pre\n', df_pre)

                df_post = df.loc[:'2',cols].tail(25)
                
                print('df_post\n', df_post)

                df = df[df['cat'].between('1','2')]

                print(df)

                df = pd.concat([df_pre, df, df_post], axis =0)

                df = df.sort_values(by=['utc'], ascending=True)

                print(df)

                #load_hdf = load_hdf.sort_values(by=['utc'], ascending = True)

                return df

        #self.df = pre_post_EPB(df)

        #self.df['utc'] = pd.to_datetime(self.df['utc']).dt.time
        self.df = self.df.drop_duplicates().dropna()
        self.df = self.df.reset_index().drop(columns=['index'], axis=1)

        df = self.df

        return df
    
# select_date = None
select_date = '2016-04-01'
w = WrangleData.frompath(path, select_date)

sat = 'A'
#sat = None
#start_time, end_time = '20:01:00', '20:10:00'
#start_lat, end_lat = -90, 90
start_lat, end_lat = -90, 90
#epb_only = False
start_time, end_time = '23:42:37','23:55:36'
cleaned_df = w.transform_EPB(sat, start_time, end_time, start_lat, end_lat)

print('Outside Class\n', cleaned_df)
#print(cleaned_df['epb'].value_counts(sort=True))
#print(cleaned_df['b_ind'].value_counts(sort=True))

#Export
# from datetime import date
# today =  str(date.today())
# joined_output = hdf_path + 'wrangled-EPB-'+ today +'.h5'
# cleaned_df.to_hdf(joined_output, key = 'efi_data', mode = 'w')
# print('Exported Wrangled EPB data')


class PlotEPB():

    def __init__(self, df):
        self.df = df
        self.p_num = 10
        #self.df = self.df[self.df['pass'] == self.p_num]
        #self.df = self.df[self.df['utc'].between('21:30:00','21:45:59')]
        self.df = self.df[self.df['lat'].between(-15,15)]
        #self.df = self.df[self.df['b_ind']!= -1] #remove non-useful data
        #self.df = self.df[self.df['long'].between(10,180)] #remove the SSA
        self.df = self.df[~self.df['mlt'].between(6,18)] #Nightime only
        #self.df = self.df[self.df['b_ind'] != 1]
        print(self.df)

        #export_path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Non-Flight Data/Analysis/Jan-22/data/systematic/nominal/'
        #export = export_path + '20160402_p10.csv'
        #self.df.to_csv(export, index=False, header = True)
        #print('data exported')

    def plotNoStdDev(self):
        

        figs, axs = plt.subplots(ncols=1, nrows=6, figsize=(10,7), 
        dpi=90, sharex=True) #3.5 for single, #5.5 for double
        axs = axs.flatten()

        x = 'lat'
        #palette_ne, palette_ti, palette_pot = 'Set1', 'Set2', 'tab10'
        palette_ne, palette_ti, palette_pot = 'flag', 'flag', 'flag'
        hue = 's_id'
        sns.lineplot(ax = axs[0], data = self.df, x = x, y ='b_ind', 
                palette = 'bone',hue = hue, legend=False)

        sns.lineplot(ax = axs[1], data = self.df, x =x, y ='Ne',
                palette = 'bone', hue = hue, legend=False)

        sns.lineplot(ax = axs[2], data = self.df, x = x, y ='b_mu', 
                #marker = 'o', linestyle='', err_style='bars', 
                palette = palette_ne, hue = hue, legend = False)

        sns.lineplot(ax = axs[3], data = self.df, x = x, y ='nec_x', 
                palette = palette_ti, hue = hue, legend = False)

        sns.lineplot(ax = axs[4], data = self.df, x = x, y ='nec_y', 
                palette = palette_pot, hue = hue, legend = False)
        
        sns.lineplot(ax = axs[5], data = self.df, x = x, y ='nec_z', 
                palette = 'Dark2', hue = hue, legend = False)
        
        ax6 = axs[5].twinx()
        sns.lineplot(ax = ax6, data = self.df, x = x, y ='nec_z', 
                palette = 'tab20b', hue = hue, legend = False)

        date_s = self.df['date'].iloc[0]
        date_e = self.df['date'].iloc[-1]
        utc_s = self.df['utc'].iloc[0]
        utc_e = self.df['utc'].iloc[-1]
      
        lat_s = self.df['lat'].iloc[0]
        lat_e = self.df['lat'].iloc[-1]

        epb_len = (lat_s - lat_e) * 110
        epb_len = "{:.0f}".format(epb_len)
        
        #print(epb_len)
        #axs[0].set_title(f'Equatorial Plasma Bubble: from {date_s} at {utc_s} to {date_e} at {utc_e}', fontsize = 11)

        epb_check = self.df['b_ind'].sum()
        if epb_check > 0:
            title = 'Equatorial Plasma Bubble'
        else:
            title = 'Quiet Period'


        axs[0].set_title(f'{title}: from {date_s} at {utc_s} ' 
                f'to {date_e} at {utc_e}. Spacecraft: {sat}, Pass: '
                f'{self.p_num}', fontsize = 11)
        axs[0].set_ylabel('EPB \n (SWARM)')
        #axs[0].set_ylim(0, 1)
        axs[0].tick_params(bottom = False)
        #axs[0].axhline( y=0.9, ls='-.', c='k')

        axs[1].set_ylabel('EPB  \n (MSSL)')
        axs[1].tick_params(bottom = False)

        #left, bottom, width, height = (1, 0, 14, 7)
        #axs[4].add_patch(Rectangle((left, bottom),width, height, alpha=1, facecolor='none'))

        '''
        axs[2].set_yscale('log')
        den = r'cm$^{-3}$'
        axs[2].set_ylabel(f'Ne ({den})')
        axs[2].tick_params(bottom = False)
        

        axs[3].set_ylabel('Ti (K)')
        axs[3].tick_params(bottom = False)

        axs[4].set_ylabel('Pot (V)')
        axs[4].tick_params(bottom = False)
        #axs[4].set_xlabel(' ')


        axs[5].set_xlabel('Latitude')
        axs[5].set_ylabel('UTC')
        axs[5].tick_params(left = False)
        ax6.set_ylabel('MLT')
        n = len(self.df) // 3.5
        [l.set_visible(False) for (i,l) in 
                enumerate(axs[5].yaxis.get_ticklabels()) if i % n != 0]'''


        ax = plt.gca()
        ax.invert_xaxis()

        plt.tight_layout()
        plt.show()

    def plotStdDev(self):

        figs, axs = plt.subplots(ncols=1, nrows=8, figsize=(10,7), 
            dpi=90, sharex=True) #3.5 for single, #5.5 for double
        axs = axs.flatten()

        x = 'utc'
        palette_ne, palette_ti, palette_pot = 'Set1', 'Set2', 'tab10'
        hue = 's_id'
        sns.lineplot(ax = axs[0], data = self.df, x = x, y ='b_ind', 
                palette = 'bone_r', hue = hue, legend =False)
        sns.lineplot(ax = axs[1], data = self.df, x = x, y ='Ne', 
                palette = palette_ne, hue = hue, legend = False)
        sns.lineplot(ax = axs[2], data = self.df, x = x, y ='Ne_std', 
                palette = palette_ne, hue = hue, legend = False)
        sns.lineplot(ax = axs[3], data = self.df, x = x, y ='Ti', 
                palette = palette_ti, hue = hue, legend = False)
        sns.lineplot(ax = axs[4], data = self.df, x = x, y ='Ti_std', 
                palette = palette_ti, hue = hue,legend = False) 
        sns.lineplot(ax = axs[5], data = self.df, x = x, y ='pot', 
                palette = palette_pot, hue = hue, legend = False)
        sns.lineplot(ax = axs[6], data = self.df, x = x, y ='pot_std', 
                palette = palette_pot, hue = hue,legend = False, markers=False)
        sns.lineplot(ax = axs[7], data = self.df, x = x, y ='epb', 
                palette = 'bone_r', hue = hue,legend = False)

        date_s = self.df['date'].iloc[0]
        date_e = self.df['date'].iloc[-1]
        utc_s = self.df['utc'].iloc[0]
        utc_e = self.df['utc'].iloc[-1]
      
        lat_s = self.df['lat'].iloc[0]
        lat_e = self.df['lat'].iloc[-1]

        epb_len = (lat_s - lat_e) * 110
        epb_len = "{:.0f}".format(epb_len)
        
        
        
        #print(epb_len)
        #axs[0].set_title(f'Equatorial Plasma Bubble: from {date_s} at {utc_s} to {date_e} at {utc_e}', fontsize = 11)

        epb_check = self.df['epb'].sum()
        if epb_check > 0:
            title = 'Equatorial Plasma Bubble'
        else:
            title = 'Quiet Period'

        

        axs[0].set_title(f'{title}: from {date_s} at {utc_s} ' 
                f'to {date_e} at {utc_e} \n size: ~{epb_len} km', fontsize = 11)
        axs[0].set_ylabel('EPB Prob')
        axs[0].set_ylim(0, 1)
        axs[0].tick_params(bottom = False)
        #axs[0].axhline( y=0.9, ls='-.', c='k')

        #left, bottom, width, height = (1, 0, 14, 7)
        #axs[4].add_patch(Rectangle((left, bottom),width, height, alpha=1, facecolor='none'))

        axs[1].set_yscale('log')
        den = r'cm$^{-3}$'
        axs[1].set_ylabel(f'Ne ({den})')
        axs[1].tick_params(bottom = False)
        #axs[1].axhline( y=60000, ls='-.', c='k')

        #axs[2].set_yscale('log')
        #axs[2].set_ylabel('Ti (K)')
        axs[2].set_ylabel('Ne \n stddev')
        #axs[2].axhline( y=950, ls='-.', c='k')
        #axs[2].set_ylim(0,1)
        axs[2].tick_params(bottom = False)

        axs[3].set_ylabel('Ti (K)')
        axs[3].tick_params(bottom = False)


        axs[4].set_ylabel('Ti \n stddev')
        #axs[4].set_ylim(0,1)
        axs[4].tick_params(bottom = False)
        #axs[4].set_xlabel(' ')
        #axs[4].legend(loc="center left", title="Sat")

        axs[5].set_ylabel('Pot (V)')
        axs[5].tick_params(bottom = False)

        axs[6].set_ylabel('Pot \n stddev')
        #axs[6].set_ylim(0,0.5)
        axs[6].tick_params(bottom = False)


        axs[7].set_ylabel('IPB Prob \n proposed')
        #axs[7].invert_xaxis()

        n = len(self.df) // 8
        #n = 50  # Keeps every 7th label
        [l.set_visible(False) for (i,l) in 
                enumerate(axs[7].xaxis.get_ticklabels()) if i % n != 0]
        #axs[4].tick_params(axis='x',labelrotation=90)
        #ax[0].set_xticks[]
        #axs[0].set_xticks([], minor=False)

        #for tic in axs[4].xaxis.get_major_ticks():
        #    tic.tick1On = tic.tick2On = True

        plt.tight_layout()
        plt.show()

    def plotRaw(self):

        figs, axs = plt.subplots(ncols=1, nrows=4, figsize=(10,7), 
            dpi=90, sharex=True) #3.5 for single, #5.5 for double
        axs = axs.flatten()

        x = 'lat'
        palette_ne, palette_ti, palette_pot = 'Set1', 'Set2', 'tab10'
        hue = 's_id'
        #sns.lineplot(ax = axs[0], data = self.df, x = x, y ='b_ind', 
        #        palette = 'bone_r', hue = hue, legend =False)
        sns.lineplot(ax = axs[0], data = self.df, x = x, y ='Ne', 
                palette = palette_ne, hue = hue, legend = False)
        sns.lineplot(ax = axs[1], data = self.df, x = x, y ='Ti', 
                palette = palette_ti, hue = hue, legend = False)
        sns.lineplot(ax = axs[2], data = self.df, x = x, y ='pot', 
                palette = palette_pot, hue = hue, legend = False)

        ax2 = axs[3].twinx()
        sns.lineplot(ax = axs[3], data = self.df, x = x, y ='mlt', 
                palette = 'bone_r', hue = hue,legend = False)
        sns.lineplot(ax = ax2, data = self.df, x = x, y ='utc', 
                palette = 'bone_r', hue = hue,legend = False)

        date_s = self.df['date'].iloc[0]
        date_e = self.df['date'].iloc[-1]
        utc_s = self.df['utc'].iloc[0]
        utc_e = self.df['utc'].iloc[-1]
      
        lat_s = self.df['lat'].iloc[0]
        lat_e = self.df['lat'].iloc[-1]

        epb_len = (lat_s - lat_e) * 110
        epb_len = "{:.0f}".format(epb_len)
        
        
        
        #print(epb_len)
        #axs[0].set_title(f'Equatorial Plasma Bubble: from {date_s} at {utc_s} to {date_e} at {utc_e}', fontsize = 11)

        epb_check = self.df['epb'].sum()
        if epb_check > 0:
            title = 'Equatorial Plasma Bubble'
        else:
            title = 'Quiet Period'

        

        axs[0].set_title(f'{title}: from {date_s} at {utc_s} ' 
                f'to {date_e} at {utc_e} \n size: ~{epb_len} km', fontsize = 11)
        #axs[0].set_ylabel('EPB Prob')
        #axs[0].set_ylim(0, 1)
        #axs[0].tick_params(bottom = False)
        #axs[0].axhline( y=0.9, ls='-.', c='k')

        #left, bottom, width, height = (1, 0, 14, 7)
        #axs[4].add_patch(Rectangle((left, bottom),width, height, alpha=1, facecolor='none'))

        axs[0].set_yscale('log')
        den = r'cm$^{-3}$'
        axs[0].set_ylabel(f'Ne ({den})')
        axs[0].tick_params(bottom = False)
        #axs[1].axhline( y=60000, ls='-.', c='k')

        #axs[2].set_yscale('log')
        #axs[2].set_ylabel('Ti (K)')
        #axs[2].set_ylabel('Ne \n stddev')
        #axs[2].axhline( y=950, ls='-.', c='k')
        #axs[2].set_ylim(0,1)
        #axs[2].tick_params(bottom = False)

        axs[1].set_ylabel('Ti (K)')
        axs[1].tick_params(bottom = False)

        #axs[4].set_ylabel('Ti \n stddev')
        #axs[4].set_ylim(0,1)
        #axs[4].tick_params(bottom = False)
        #axs[4].set_xlabel(' ')
        #axs[4].legend(loc="center left", title="Sat")

        axs[2].set_ylabel('Pot (V)')
        axs[2].tick_params(bottom = False)

        #axs[6].set_ylabel('Pot \n stddev')
        #axs[6].set_ylim(0,0.5)
        #axs[6].tick_params(bottom = False)


        axs[3].set_ylabel('MLT')
        ax2.set_ylabel('utc')
        #axs[7].invert_xaxis()

        #n = len(self.df) // 8
        #n = 50  # Keeps every 7th label
        #[l.set_visible(False) for (i,l) in 
        #        enumerate(axs[7].xaxis.get_ticklabels()) if i % n != 0]
        #axs[4].tick_params(axis='x',labelrotation=90)
        #ax[0].set_xticks[]
        #axs[0].set_xticks([], minor=False)

        #for tic in axs[4].xaxis.get_major_ticks():
        #    tic.tick1On = tic.tick2On = True

        plt.tight_layout()
        plt.show()

    def plotEPBCount(self):
        #self.df == self.df
        print(self.df)

        figs, axs = plt.subplots(ncols=2, nrows=1, figsize=(9.5, 4), 
                sharey=True)
        axs = axs.flatten()

        self.df.plot(kind="scatter", x="long", y="lat", alpha=0.4, ax=axs[0],
                c="epb", cmap=plt.get_cmap("jet"), colorbar=True)

        self.df.plot(kind="scatter", x="long", y="lat", alpha=0.4, ax=axs[1],
                c="b_ind", cmap=plt.get_cmap("jet"), colorbar=True)

        plt.legend()
        plt.show()


p = PlotEPB(cleaned_df)
panels = p.plotNoStdDev()
#panels = p.plotStdDev()
#panels = p.plotRaw()
#counts = p.plotEPBCount()
