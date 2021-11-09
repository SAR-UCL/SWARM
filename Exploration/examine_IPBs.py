
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import seaborn as sns
from datetime import date


#Loading and exporting
path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Non-Flight Data/Analysis/Nov-21/data/April-16/'
today =  str(date.today())
file_name = 'joined-data-'+ today +'.h5'
load_hdf = path + file_name
load_hdf = pd.read_hdf(load_hdf)

load_hdf = load_hdf[load_hdf['date'] == '2016-04-04']
#load_hdf = load_hdf[load_hdf['b_ind'] == 1]
#load_hdf = load_hdf[load_hdf['b_prob'] >= 0.85]
#load_hdf = load_hdf[load_hdf['b_prob'].between(0.8, 0.9)]
#print(load_hdf)


#load_hdf = load_hdf[load_hdf['utc'].between('00:36', '00:53')] #2016-04-03. This is in the supervisor presi on 17-11-21
#load_hdf = load_hdf[load_hdf['utc'].between('00:40', '00:50')] #2016-04-03. This is in the supervisor presi on 17-11-21
#load_hdf = load_hdf[load_hdf['utc'].between('1z:20', '19:34')] #2016-04-03b. This is in the supervisor presi on 17-11-21
#load_hdf = load_hdf[load_hdf['lat'].between(-8, 28)] #2016-04-03b. This is in the supervisor presi on 17-11-21
load_hdf = load_hdf[load_hdf['utc'].between("20:18", "20:26")] #2016-04-04. 
#load_hdf = load_hdf[load_hdf['utc'].between('21:30', '21:45:30')] #2017-04-17
#load_hdf = load_hdf[load_hdf['utc'].between('21:13', '21:25')] #2016-04-05
#load_hdf = load_hdf[load_hdf['utc'].between('00:06', '00:11:40')] #2019-03-16 slim
#load_hdf = load_hdf[load_hdf['utc'].between('00:04', '00:12:20')] #2019-03-16
#load_hdf = load_hdf[load_hdf['utc'].between('00:02', '00:14')] #2019-03-016 (wider)

#load_hdf = load_hdf[load_hdf['mlt'].between(0,6)]
#load_hdf = load_hdf[load_hdf['lat'].between(-20,20)]
#load_hdf = load_hdf[load_hdf['Ne_c'] <= -0.4]

#load_hdf = load_hdf[::120]
load_hdf = load_hdf.sort_values(by=['utc'], ascending = True)
load_hdf = load_hdf[load_hdf['s_id'] == "A"]
print(load_hdf)
#print(load_hdf.dtypes)



figs, axs = plt.subplots(ncols=1, nrows=7, figsize=(10,8.5), dpi=85, sharex=True) #3.5 for single, #5.5 for double
axs = axs.flatten()

x = 'utc'
#sns.set_palette("Reds")
palette, hue = 'rocket', 's_id'
sns.lineplot(ax = axs[0], data = load_hdf, x = x, y ='b_prob', palette = 'bone_r', hue = hue)
sns.lineplot(ax = axs[1], data = load_hdf, x = x, y ='Ne', palette = palette, hue = hue, legend = False)
sns.lineplot(ax = axs[2], data = load_hdf, x = x, y ='Ne_std5', palette = palette, hue = hue, legend = False)
sns.lineplot(ax = axs[3], data = load_hdf, x = x, y ='Ti', palette = palette, hue = hue, legend = False)
sns.lineplot(ax = axs[4], data = load_hdf, x = x, y ='Ti_std5', palette = palette, hue = hue)
sns.lineplot(ax = axs[5], data = load_hdf, x = x, y ='pot', palette = palette, hue = hue, legend = False)
sns.lineplot(ax = axs[6], data = load_hdf, x = x, y ='pot_std5', palette = palette, hue = hue)

date_s = load_hdf['date'].iloc[0]
date_e = load_hdf['date'].iloc[-1]
utc_s = load_hdf['utc'].iloc[0]
utc_e = load_hdf['utc'].iloc[-1]

axs[0].set_title(f'Equatorial Plasma Bubble: from {date_s} at {utc_s} to {date_e} at {utc_e}', fontsize = 11)
axs[0].set_ylabel('IPB Prob')
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
#axs[2].set_ylabel('Ne %/s')
#axs[2].axhline( y=950, ls='-.', c='k')

#axs[3].set_ylabel('Pot (V)')

#axs[4].set_ylabel('Te (K)')
axs[4].set_xlabel(' ')
axs[4].legend(loc="center left", title="Sat")

n = len(load_hdf) // 8
#n = 50  # Keeps every 7th label
[l.set_visible(False) for (i,l) in enumerate(axs[6].xaxis.get_ticklabels()) if i % n != 0]
#axs[4].tick_params(axis='x',labelrotation=90)
#ax[0].set_xticks[]
#axs[0].set_xticks([], minor=False)

#for tic in axs[4].xaxis.get_major_ticks():
#    tic.tick1On = tic.tick2On = True

plt.tight_layout()
plt.show()



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