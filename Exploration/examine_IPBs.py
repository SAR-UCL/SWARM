
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
#from matplotlib.colors import LogNorm

#Loading and exporting
path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Non-Flight Data/Analysis/Nov-21/data/'

file_name = 'joined-data_20211104.h5'
load_hdf = path + file_name
load_hdf = pd.read_hdf(load_hdf)

load_hdf = load_hdf[load_hdf['date'] == '2016-04-05']
#load_hdf = load_hdf[load_hdf['b_ind'] == 1]
#load_hdf = load_hdf[load_hdf['b_prob'] >= 0.85]
#load_hdf = load_hdf[load_hdf['b_prob'].between(0.8, 0.9)]
print(load_hdf)


#load_hdf = load_hdf[load_hdf['utc'].between('00:36', '00:53')] #2016-04-03. This is in the supervisor presi on 17-11-21
#load_hdf = load_hdf[load_hdf['utc'].between('19:20', '19:32')] #2016-04-03b. This is in the supervisor presi on 17-11-21
#load_hdf = load_hdf[load_hdf['utc'].between('21:30', '21:45:30')] #2017-04-17
load_hdf = load_hdf[load_hdf['utc'].between('21:13', '21:25')] #2016-04-05

#load_hdf = load_hdf.drop_duplicates(subset=['utc'])
#load_hdf = load_hdf[::20]

#load_hdf = load_hdf[::120]
load_hdf = load_hdf.sort_values(by=['utc'], ascending = True)
print(load_hdf)


figs, axs = plt.subplots(ncols=1, nrows=5, figsize=(10,6.5), dpi=85, sharex=True) #3.5 for single, #5.5 for double
axs = axs.flatten()

x = 'utc'

sns.lineplot(ax = axs[0], data = load_hdf, x = x, y ='b_prob')
sns.scatterplot(ax = axs[1], data = load_hdf, x = x, y ='Ne')
sns.scatterplot(ax = axs[2], data = load_hdf, x = x, y ='Ti')
sns.scatterplot(ax = axs[3], data = load_hdf, x = x, y ='pot')
sns.scatterplot(ax = axs[4], data = load_hdf, x = x, y ='Te')

date = load_hdf['date'].iloc[0]
axs[0].set_title(f'Equatorial Plasma Bubble: {date}', fontsize = 11)
axs[0].set_ylabel('IPB Prob')
axs[0].set_ylim(0, 1)
axs[0].axhline( y=0.9, ls='-.', c='k')

axs[1].set_yscale('log')
den = r'cm$^{-3}$'
axs[1].set_ylabel(f'Ne ({den})')
#axs[1].axhline( y=60000, ls='-.', c='k')

axs[2].set_ylabel('Ti (K)')
#axs[2].axhline( y=950, ls='-.', c='k')

axs[3].set_ylabel('Pot (V)')

axs[4].set_ylabel('Te (K)')
axs[4].set_xlabel(' ')

n = 100  # Keeps every 7th label
[l.set_visible(False) for (i,l) in enumerate(axs[4].xaxis.get_ticklabels()) if i % n != 0]
#axs[4].tick_params(axis='x',labelrotation=90)
#ax[0].set_xticks[]
#axs[0].set_xticks([], minor=False)

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
plt.show()'''