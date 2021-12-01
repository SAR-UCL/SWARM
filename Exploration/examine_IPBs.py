
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
pd.set_option('display.max_rows', 10) #or 10 or None
import seaborn as sns
from datetime import date


#Load exported .hdf files
hdf_path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Non-Flight Data/Analysis/Dec-21/data/April-16/'

class WrangleData():
    
    def __init__(self, pathfile, select_date=None):
        df = pd.read_hdf(pathfile)
        self.df = df if select_date is None else df[df['date'] == select_date]

    @classmethod  # method that can be called before obj is created
    def frompath(cls, hdf_path, select_date=None):
        today =  str(date.today())
        file_name = 'joined-data-'+ today +'.h5'
        return cls(hdf_path+file_name, select_date)

    def classify_EPB(self, ne_std, b_ind, ti_std, pot_std):
        #if ne_std > 0.095 or b_ind == 1:
        if ne_std > 0.01 and ti_std > 0.01 and pot_std > 0.01:
            return 1
        else:
            return 0

    def transform_EPB(self, sat, s_time, e_time):
        #Filter the data
        self.df = self.df[self.df['b_ind']!= -1] #remove non-useful data
        self.df = self.df[self.df['long'].between(10,180)] #remove the SSA
        self.df = self.df[~self.df['mlt'].between(6,18)] #Nightime only
        self.df = self.df[self.df['s_id'] == sat]
        self.df = self.df[self.df['utc'].between(s_time, e_time)]
        self.df = self.df[self.df['lat'].between(-30,30)] #EPB region 

        #Call classifier function
        self.new_classifier()
     
    def new_classifier(self):

        
        #New EPB classification
        self.df['epb'] = self.df.apply(lambda x: self.classify_EPB(x.Ne_std, 
                x.b_ind, x.Ti_std, x.pot_std), axis=1)
        #self.df = self.df[self.df['epb'] == 1]
        
        
        #Calculate EPB start, middle and end
        epb_cat = 'epb'
        self.df['change'] = self.df[epb_cat] - self.df[epb_cat].shift(1)
        self.df = self.df.replace({'change':{-1:2}})
        self.df['temp'] = np.where((self.df[epb_cat] == 1) 
                & (self.df['change'] == 0), 1.5, 0)

        #Reassign s, m & e to one column
        #https://stackoverflow.com/questions/39109045/
        #numpy-where-with-multiple-conditions
        conditions = [self.df['change'] == 1, self.df['temp'] == 1.5, 
                 self.df['change'] ==2]
        output = [1,1.5,2]
        self.df['cat'] = np.select(conditions, output, default = '0')

        # if epb_only == True:
        #     self.df = self.df[self.df['cat'].between('1','2')]
        # else:
        #     pass

        self.df = self.df.reset_index().drop(columns=['change',
                'temp','index'], axis=1)
        

        #Determine range
        def pre_post_EPB(df):


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

        #self.df = pre_post_EPB(self.df)

        self.df = self.df.drop_duplicates().dropna()
        self.df = self.df.reset_index().drop(columns=['index'], axis=1)

        print(self.df)
        return self.df
    

select_date = None
select_date = '2016-04-07'
w = WrangleData.frompath(hdf_path, select_date)

sat = 'A'
start_time = '09:05:32'
end_time = '09:20:32'
epb_only = False
#start_time = '00:00:00'
#end_time = '23:59:59'
cleaned_df = w.transform_EPB(sat, start_time, end_time)
#print(cleaned_df)


class PlotEPB():

    def __init__(self, df):
        self.df = df

    def plotPanels(self):

        figs, axs = plt.subplots(ncols=1, nrows=8, figsize=(10,7), 
            dpi=90, sharex=True) #3.5 for single, #5.5 for double
        axs = axs.flatten()

        x = 'lat'
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

        

        axs[0].set_title(f'{title}: from {date_s} at {utc_s}' 
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
panels = p.plotPanels()
#counts = p.plotEPBCount()

#select_date = '2016-04-01'
#w = wrangler.frompath(hdf_path, select_date)

#p = plot()
#plot_panels = p.plotPanels()
#prepplots(w)

# df = w.filter()
# df['epb'] = df.apply(lambda x: w.classifyEPB(x.Ne_std, x.b_ind), axis=1)
# print(df)

#print('classy \n',filt)



def tempCats(x, y):
    if x > 0.095:
        return 1
    else:
        return 0

#joined_cdf['temp_prob'] = joined_cdf.apply(lambda x: tempCats(x.Ne_std, x.pot_std), axis=1)

def newEPB(x, y):
    import numpy as np
    if x or y == np.abs(1):
        return 1
    else:
        return 0

#joined_cdf['n_prob'] = joined_cdf.apply(lambda x: newEPB(x.b_ind, x.temp_prob), axis=1)


def removeSSA(df):
    df = df[df['long'].between(10,180)]
    return df
#load_hdf = removeSSA(load_hdf)

def dayNight(df):
    df = df[~df['mlt'].between(6,18)]
    return df
#load_hdf = dayNight(load_hdf)

#load_hdf = load_hdf[load_hdf['b_ind'] == 0]
#load_hdf = load_hdf[load_hdf['b_prob'] >= 0.85]
#load_hdf = load_hdf[load_hdf['b_prob'].between(0.8, 0.9)]
#print(load_hdf)
'''

#load_hdf = load_hdf[load_hdf['utc'].between('00:36', '00:53')] #2016-04-03. This is in the supervisor presi on 17-11-21
#load_hdf = load_hdf[load_hdf['utc'].between('00:40', '00:50')] #2016-04-03. This is in the supervisor presi on 17-11-21
#load_hdf = load_hdf[load_hdf['utc'].between('1z:20', '19:34')] #2016-04-03b. This is in the supervisor presi on 17-11-21
#load_hdf = load_hdf[load_hdf['lat'].between(-8, 28)] #2016-04-03b. This is in the supervisor presi on 17-11-21
#load_hdf = load_hdf[load_hdf['utc'].between("20:18", "20:26")] #2016-04-04. 
#load_hdf = load_hdf[load_hdf['utc'].between('21:30', '21:45:30')] #2017-04-07
#load_hdf = load_hdf[load_hdf['utc'].between('21:13', '21:25')] #2016-04-05

#Swarm C
#load_hdf = load_hdf[load_hdf['utc'].between('00:06', '00:11:40')] #2019-03-16 slim
#load_hdf = load_hdf[load_hdf['utc'].between('00:04', '00:12:20')] #2019-03-16
#load_hdf = load_hdf[load_hdf['utc'].between('00:02', '00:14')] #2019-03-016 (wider)

#Not an EPB
#load_hdf = load_hdf[load_hdf['utc'].between('02:54', '03:10')] #2016-04-07

#Swarm A
#load_hdf = load_hdf[load_hdf['utc'].between('00:01:30', '00:12')] #2019-03-16

#load_hdf = load_hdf[load_hdf['mlt'].between(0,6)]
load_hdf = load_hdf[load_hdf['lat'].between(-30,30)]
#load_hdf = load_hdf[load_hdf['Ne_c'] <= -0.4]

#Not an EPB
load_hdf = load_hdf[load_hdf['utc'].between('09:05:20', '09:20:50')] #2016-04-07


#load_hdf = load_hdf[::120]
load_hdf = load_hdf.sort_values(by=['utc'], ascending = True)
load_hdf = load_hdf[load_hdf['s_id'] == "A"]
print(load_hdf)
#print(load_hdf.dtypes)


figs, axs = plt.subplots(ncols=1, nrows=8, figsize=(10,7), dpi=90, sharex=True) #3.5 for single, #5.5 for double
axs = axs.flatten()

x = 'lat'
#sns.set_palette("Reds")
palette_ne, palette_ti, palette_pot, hue = 'Set1', 'Set2', 'tab10', 's_id'
sns.lineplot(ax = axs[0], data = load_hdf, x = x, y ='b_prob', palette = 'bone_r', hue = hue, legend =False)
sns.lineplot(ax = axs[1], data = load_hdf, x = x, y ='Ne', palette = palette_ne, hue = hue, legend = False)
sns.lineplot(ax = axs[2], data = load_hdf, x = x, y ='Ne_std', palette = palette_ne, hue = hue, legend = False)
sns.lineplot(ax = axs[3], data = load_hdf, x = x, y ='Ti', palette = palette_ti, hue = hue, legend = False)
sns.lineplot(ax = axs[4], data = load_hdf, x = x, y ='Ti_std', palette = palette_ti, hue = hue,legend = False) 
sns.lineplot(ax = axs[5], data = load_hdf, x = x, y ='pot', palette = palette_pot, hue = hue, legend = False)
sns.lineplot(ax = axs[6], data = load_hdf, x = x, y ='pot_std', palette = palette_pot, hue = hue,legend = False)
#sns.lineplot(ax = axs[7], data = load_hdf, x = x, y ='n_prob', palette = 'bone_r', hue = hue,legend = False)

date_s = load_hdf['date'].iloc[0]
date_e = load_hdf['date'].iloc[-1]
utc_s = load_hdf['utc'].iloc[0]
utc_e = load_hdf['utc'].iloc[-1]

#axs[0].set_title(f'Equatorial Plasma Bubble: from {date_s} at {utc_s} to {date_e} at {utc_e}', fontsize = 11)
axs[0].set_title(f'Quiet Transit: from {date_s} at {utc_s} to {date_e} at {utc_e}', fontsize = 11)
axs[0].set_ylabel('EPB Prob')
axs[0].set_ylim(0, 1)
#axs[0].axhline( y=0.9, ls='-.', c='k')

#left, bottom, width, height = (1, 0, 14, 7)
#axs[4].add_patch(Rectangle((left, bottom),width, height, alpha=1, facecolor='none'))


axs[1].set_yscale('log')
den = r'cm$^{-3}$'
axs[1].set_ylabel(f'Ne ({den})')
#axs[1].axhline( y=60000, ls='-.', c='k')

#axs[2].set_yscale('log')
#axs[2].set_ylabel('Ti (K)')
axs[2].set_ylabel('Ne \n stddev')
#axs[2].axhline( y=950, ls='-.', c='k')
axs[2].set_ylim(0,1)

axs[3].set_ylabel('Ti (K)')


axs[4].set_ylabel('Ti \n stddev')
axs[4].set_ylim(0,1)
#axs[4].set_xlabel(' ')
#axs[4].legend(loc="center left", title="Sat")

axs[5].set_ylabel('Pot (V)')
axs[6].set_ylabel('Pot \n stddev')
axs[6].set_ylim(0,0.5)


axs[7].set_ylabel('IPB Prob \n proposed')

#n = len(load_hdf) // 8
#n = 50  # Keeps every 7th label
#[l.set_visible(False) for (i,l) in enumerate(axs[7].xaxis.get_ticklabels()) if i % n != 0]
#axs[4].tick_params(axis='x',labelrotation=90)
#ax[0].set_xticks[]
#axs[0].set_xticks([], minor=False)

#for tic in axs[4].xaxis.get_major_ticks():
#    tic.tick1On = tic.tick2On = True

plt.tight_layout()
plt.show()
'''

#Single Plot
'''
plt.figure(dpi = 90, figsize=(12,4))
sns.scatterplot(data = load_hdf, x = 'utc', y ='Ne')
#plt.xticks([], [])
plt.yscale('log')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
'''