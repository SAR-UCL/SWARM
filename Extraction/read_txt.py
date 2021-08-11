import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import numpy as np


#path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Data_old/20180622/20180622.txt'
#path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Data/20180622/20180622.txt'
path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Data/Dayside/20180525/Level2/20180525_IPD.txt'
csv_output = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Data/Dayside/20180525/Level2/'

read_swarm = pd.read_csv(path, delim_whitespace=True, skiprows=38,  dtype={"RODI10s":"string","RODI20s": "string"}) #65 for EFIATIE, #38 for IPD
#print(read_swarm)

def IPD():
    read_IPD = read_swarm[['Record','Timestamp','Latitude','Longitude','Ne','Te']].reset_index().drop(columns=['index'])
    read_IPD['Latitude'] = read_IPD.apply(lambda x: x['Latitude'][:5], axis = 1)
    read_IPD['Longitude'] = read_IPD.apply(lambda x: x['Latitude'][:5], axis = 1)
    read_IPD['Ne'] = read_IPD['Ne'] * 1e6
    read_IPD = read_IPD.loc[read_IPD['Ne'] < 2e+11]
    #read_IPD = read_IPD.loc[read_IPD['Te'] < 3000]
    read_IPD =  read_IPD[read_IPD['Te'].between(500,3000)]
    read_IPD =  read_IPD[read_IPD['Latitude'].between('-50','50')]

    def select_hours(df):
        df['hr'] = pd.to_datetime(df['Timestamp']).dt.hour
        df.drop_duplicates(subset=['Record', 'hr'], keep='first', inplace=True)
        df.drop('hr', axis=1, inplace=True)
        return df

    #plot_IPD = read_IPD
    plot_IPD = select_hours(read_IPD).reset_index().drop(columns=['index'])
    plot_IPD['Time_short'] = plot_IPD.apply(lambda x: x['Timestamp'][:-10], axis = 1)
    print(plot_IPD)

    x_axis = 'Time_short'
    figs, axs = plt.subplots(ncols=2, nrows=1, figsize=(8.5,3.5), sharex=True)
    axs = axs.flatten()
    sns.scatterplot(data = plot_IPD, ax = axs[0], x = x_axis, y = 'Ne')
    sns.scatterplot(data = plot_IPD, ax = axs[1], x = x_axis, y = 'Te')

    axs[0].xaxis.set_major_locator(plt.MaxNLocator(10))
    axs[1].xaxis.set_major_locator(plt.MaxNLocator(10))

    axs[0].set_yscale('log')

    plt.tight_layout()
    plt.show()

#IPD()


#read_swarm['Height'] = read_swarm['Height'] / 1000

#read_swarm = read_swarm.drop(columns=['Radius','Tn_msis','Flag_ti_meas','Flag_ti_model','QDLatitude'])

#read_swarm =  read_swarm[read_swarm['Latitude'].between(-50,50)]
#read_swarm =  read_swarm[read_swarm['MLT'].between(11,12.30)]
#read_swarm['utc'] = pd.to_datetime(read_swarm['Timestamp'],format='%H:%M:%S.%f').dt.time

#Select one hour per date
def select_hours(df):
  df['hr'] = pd.to_datetime(df['Timestamp']).dt.hour
  df.drop_duplicates(subset=['Record', 'hr'], keep='first', inplace=True)
  df.drop('hr', axis=1, inplace=True)
  return df

#plot_fits = select_hours(read_swarm).reset_index().drop(columns=['index'])
#plot_fits = read_swarm.reset_index().drop(columns=['index'])

#plot_fits['Time_short'] = plot_fits.apply(lambda x: x['Timestamp'][:-10], axis = 1)
#plot_fits['Timestamp'] = plot_fits['Timestamp'].dt.time("%H")

#csv_path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/Phoenix/Instrument Data/analysis/Aug-21/data/'
#csv_output_pathfile = csv_output + '20180525_IPD.csv' # -4 removes '.pkts' or '.dat'
#read_swarm.to_csv(csv_output_pathfile, index = False, header = True)



#print(read_swarm)

'''
#print(read_swarm)
hue, palette, style = 'Record', None, None
x_axis = 'Time_short'
figs, axs = plt.subplots(ncols=2, nrows=2, figsize=(8.5,5.5), sharex=True)
axs = axs.flatten()
sns.scatterplot(data = plot_fits, ax = axs[0], x = x_axis, y = 'Te_adj_LP', palette=palette, hue = hue, style = style)
sns.scatterplot(data = plot_fits, ax = axs[1], x = x_axis, y = 'Ti_meas_drift', palette=palette, hue = hue, style = style)
sns.scatterplot(data = plot_fits, ax = axs[2], x = x_axis, y = 'Ti_model_drift', palette=palette, hue = hue, style = style)
sns.scatterplot(data = plot_fits, ax = axs[3], x = x_axis, y = 'Height', palette=palette, hue = hue, style = style)


axs[0].set_ylabel('Electron Temp (LP) [K]')
axs[1].set_ylabel('Estimated Ion \n Temp (TII Model) [K]')
axs[2].set_ylabel('Estimated Ion \n Temp (Weimer 2005) [K]')
axs[3].set_ylabel('Height [km]')

#n = 2
#[l.set_visible(False) for (i,l) in enumerate(axs.xaxis.get_ticklabels()) if i % n != 0]

axs[2].xaxis.set_major_locator(plt.MaxNLocator(10))
axs[3].xaxis.set_major_locator(plt.MaxNLocator(10))

#axs[2].tick_params(axis='x',labelrotation=90)
#axs[3].tick_params(axis='x',labelrotation=90)

plt.tight_layout()
plt.show()'''
