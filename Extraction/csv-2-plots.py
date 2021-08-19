import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import numpy as np

def joinIPDEFI():
    IPD_path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Data/Dayside/All/IPD-EFI/IPD-Dayside.csv'
    EFI_path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Data/Dayside/All/IPD-EFI/EFI-Dayside.csv'

    csv_output_path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Data/Dayside/All/IPD-EFI/'

    read_IPD = pd.read_csv(IPD_path)
    
    read_EFI = pd.read_csv(EFI_path)
    read_EFI = read_EFI = read_EFI.drop(columns=['Altitude','Latitude','Longitude'])


    #print(read_EFI)
    #print(read_IPD)

    joined_IPDEFI = pd.merge(read_IPD, read_EFI,  how='left', left_on=['Date','UTC'], right_on = ['Date','UTC'])
    #merged_inner = pd.merge(left=survey_sub, right=species_sub, left_on='species_id', right_on='species_id')
    #joined_IPDEFIT = joined_IPDEFI.at[29844866, 'name']
    joined_IPDEFI = joined_IPDEFI.loc[joined_IPDEFI['Date'] == '2018-05-23']

    #print(joined_IPDEFI)
    
    joined_IPDEFI['Ne'] = joined_IPDEFI['Ne'] * 1e6
    #joined_IPDEFI = joined_IPDEFI[joined_IPDEFI['UTC'].between("11:31:07.197000","15:15:34.197000")]
    #joined_IPDEFI = joined_IPDEFI.loc[joined_IPDEFI['Ne'] < 2e+11]
    #joined_IPDEFI = joined_IPDEFI[joined_IPDEFI['Te_x'].between(1000,3000)]
    #joined_IPDEFI = joined_IPDEFI[joined_IPDEFI['Te_y'].between(1000,3000)]
    #joined_IPDEFI = joined_IPDEFI[joined_IPDEFI['Ti'].between(600,2000)]
    #joined_IPDEFI = joined_IPDEFI[joined_IPDEFI['TiM'].between(600,2000)]
    #joined_IPDEFI = joined_IPDEFI[joined_IPDEFI['Ti'].between(500,3000)]
    joined_IPDEFI = joined_IPDEFI[joined_IPDEFI['Latitude'].between(-49,-47)]
    joined_IPDEFI = joined_IPDEFI[joined_IPDEFI['Longitude'].between(-80,-50)] #sydney is 150, santiago is -70
    #print(joined_IPDEFI)

    
    def select_hours(df):
        import datetime as dt
        df['hr'] = pd.to_datetime(df['UTC']).dt.hour
        df.drop_duplicates(subset=['Date', 'hr'], keep='first', inplace=True)
        df.drop('hr', axis=1, inplace=True)
        return df


    '''
    joined_IPDEFI = select_hours(joined_IPDEFI).reset_index().drop(columns=['index'])    
    joined_IPDEFI['Time_short'] = joined_IPDEFI.apply(lambda x: x['UTC'][:-13], axis = 1)
    joined_IPDEFI['Time_short'] = (joined_IPDEFI['Time_short'].astype(int))'''
    
    print(joined_IPDEFI)

    #csv_output_pathfile = csv_output_path + "/IPD-EFI-Cleaned.csv" # -4 removes '.pkts' or '.dat'
    #joined_IPDEFI.to_csv(csv_output_pathfile, index = False, header = True)

joinIPDEFI()

def cleanIPD():
    path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Data/Dayside/All/IPD/IPD-Dayside.csv'
    #path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Data/Dayside/All/IPD/IPD-Dayside.csv'
    csv_output_path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Data/Dayside/All/IPD/'

    #read_swarm = pd.read_excel(path, dtype={"Latitude":"string","Longitude":"string","RODI10s":"string","RODI20s": "string"})
    #read_swarm = pd.read_csv(path, dtype={"Latitude":"string","Longitude":"string","RODI10s":"string","RODI20s": "string"})
    read_swarm = pd.read_csv(path)
    
    #clean_IPD = read_swarm[['Record','Timestamp','Latitude','Longitude','Ne','Te']].reset_index().drop(columns=['index'])

    #clean_IPD = clean_IPD.loc[clean_IPD['Record'] == '19/05/2018']

    #clean_IPD['Latitude'] = clean_IPD.apply(lambda x: x['Latitude'][:5], axis = 1)
    #clean_IPD['Longitude'] = clean_IPD.apply(lambda x: x['Longitude'][:5], axis = 1)

    #clean_IPD['Latitude'] = clean_IPD['Latitude'].astype(float)
    #clean_IPD['Longitude'] = clean_IPD['Longitude'].astype(float)  
    
    #print(clean_IPD)
    clean_IPD = read_swarm
    clean_IPD['Ne'] = clean_IPD['Ne'] * 1e6
    #clean_IPD = clean_IPD.loc[clean_IPD['Ne'] < 2e+11]
    clean_IPD = clean_IPD[clean_IPD['Te'].between(1000,3000)]
    #clean_IPD = clean_IPD[clean_IPD['Latitude'].between(-60,60)]
    clean_IPD = clean_IPD[clean_IPD['Longitude'].between(69,71)] #sydney is 150, santiago is -70
    #print(clean_IPD)

    
    def select_hours(df):
        import datetime as dt
        df['hr'] = pd.to_datetime(df['UTC']).dt.hour
        df.drop_duplicates(subset=['Date', 'hr'], keep='first', inplace=True)
        df.drop('hr', axis=1, inplace=True)
        return df

    clean_IPD = select_hours(clean_IPD).reset_index().drop(columns=['index'])    
    clean_IPD['Time_short'] = clean_IPD.apply(lambda x: x['UTC'][:-13], axis = 1)
    clean_IPD['Time_short'] = (clean_IPD['Time_short'].astype(int)) - 4
    
    print(clean_IPD)


    csv_output_pathfile = csv_output_path + "/IPD-Dayside-Cleaned.csv" # -4 removes '.pkts' or '.dat'
    clean_IPD.to_csv(csv_output_pathfile, index = False, header = True)

#cleanIPD()
    
def plotIPD():

    #path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Data/Dayside/All/IPD/IPD-Dayside-Cleaned.csv'
    path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Data/Dayside/All/IPD-EFI/IPD-EFI-Cleaned.csv'

    
    plot_IPD = pd.read_csv(path)
    print(plot_IPD)
    
    
    x_axis = 'Time_short'
    #plt.rcParams['font.size'] = '11.5'
    figs, axs = plt.subplots(ncols=2, nrows=2, figsize=(8.5,5.5), sharex=True) #3.5 for single, #5.5 for double
    axs = axs.flatten()

    hue = 'Date'
    sns.scatterplot(data = plot_IPD, ax = axs[0], x = x_axis, y = 'Ne', hue = hue)
    sns.scatterplot(data = plot_IPD, ax = axs[1], x = x_axis, y = 'Te_x', hue = hue)
    sns.scatterplot(data = plot_IPD, ax = axs[2], x = x_axis, y = 'Ti', hue = hue)
    sns.scatterplot(data = plot_IPD, ax = axs[3], x = x_axis, y = 'TiM', hue = hue)

    axs[0].xaxis.set_major_locator(plt.MaxNLocator(8))
    axs[1].xaxis.set_major_locator(plt.MaxNLocator(8))

    axs[0].get_legend().remove()
    axs[1].get_legend().remove()
    axs[2].get_legend().remove()
    axs[3].get_legend().remove()
    axs[3].legend(bbox_to_anchor=(1.05, 1.1), loc='upper left', prop={'size': 8.5})
    #axs[3].get_legend().remove()

    axs[0].set_yscale('log')
    #axs[1].set_yscale('log')

    plt.tight_layout()
    plt.show()

#plotIPD()

def plotSwenix():

    path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWENIX/data/densities-temps_130821.csv' 

    
    load_swenix = pd.read_csv(path)
    load_swenix = load_swenix.drop(columns=['swa_alt','notes'])
    print(load_swenix)

    load_swenix['date'] = load_swenix.apply(lambda x: x['date'][:5], axis = 1)

    melt_den = pd.melt(load_swenix, id_vars=['date','terminator'], value_vars=['swa_den','phom_den','phok_den'])
    melt_temp = pd.melt(load_swenix, id_vars=['date','terminator'], value_vars=['swa_it','phom_it','phok_it'])

    #Rename temperature columns
    melt_den = melt_den.rename(columns={"terminator": "Conditions Match"})
    melt_den = melt_den.replace({'Conditions Match"': {"yes": "Yes", "no": "No"}})
    melt_temp = melt_temp.rename(columns={"terminator": "Conditions Match"})
    melt_temp = melt_temp.replace({'Conditions Match"': {"yes": "Yes", "no": "No"}})

    #Rename density columns
    melt_den = melt_den.rename(columns={"variable": "Mission"})
    melt_den = melt_den.replace({'Mission': {"swa_den": "SWARM", "phom_den": "Phoenix (Maxwellian)", "phok_den":"Phoenix (Kappa)"}})
    melt_temp = melt_temp.rename(columns={"variable": "Mission"})
    melt_temp = melt_temp.replace({'Mission': {"swa_it": "SWARM", "phom_it": "Phoenix (Maxwellian)", "phok_it":"Phoenix (Kappa)"}})
    #melt_temp = melt_temp.loc[melt_temp['Mission'] != 'Phoenix (Kappa)']

    #print(melt_den, melt_temp)
    plt.rcParams['font.size'] = '10.5'
    x_axis = 'date'
    figs, axs = plt.subplots(ncols=2, nrows=2, figsize=(7.5,5.5), sharex=True, sharey =False) #3.5 for single, #5.5 for double
    axs = axs.flatten()

    hue = 'Conditions Match'
    #sns.scatterplot(data = melt_den, ax = axs[0], x = x_axis, y = 'value', hue = hue)
    #sns.scatterplot(data = melt_temp, ax = axs[1], x = x_axis, y = 'value', hue = hue)

    hue, style = 'Mission', 'Conditions Match'
    sns.scatterplot(data = melt_den, ax = axs[2], x = x_axis, y = 'value', hue = huey, style = hue)
    sns.scatterplot(data = melt_temp, ax = axs[3], x = x_axis, y = 'value', hue = huey, style = hue)
    
    #For CEAS Paper
    '''
    plt.figure(figsize=(5.5,3.5), dpi=90)
    plt.rcParams['font.size'] = '11.5'
    sns.scatterplot(data = melt_temp, x = 'date', y = 'value', hue = 'Terminator', style = 'Mission')
    plt.ylabel(' Temp [K]')
    plt.xlabel(' ')'''
    

    #axs[0].xaxis.set_major_locator(plt.MaxNLocator(8))
    #axs[1].xaxis.set_major_locator(plt.MaxNLocator(8))

    den = r'm$^{-3}$'
    axs[0].set_ylabel(f'Density [{den}]')
    axs[1].set_ylabel(f'Temp [K]')
    axs[2].set_ylabel(f'Density [{den}]')
    axs[3].set_ylabel(f'Temp [K]')

    #axs[2].set_xlabel(' ')
    #axs[3].set_xlabel(' ')

    axs[0].get_legend().remove()
    axs[1].get_legend().remove()
    axs[2].get_legend().remove()
    axs[3].get_legend().remove()
    axs[3].legend(bbox_to_anchor=(1.05, 1.1), loc='upper left', prop={'size': 8.5})

    axs[0].set_yscale('log')
    axs[2].set_yscale('log')
    #axs[1].set_yscale('log')

    axs[2].tick_params(axis='x',labelrotation=90)
    axs[3].tick_params(axis='x',labelrotation=90)

    plt.tight_layout()
    plt.show()

plotSwenix()
