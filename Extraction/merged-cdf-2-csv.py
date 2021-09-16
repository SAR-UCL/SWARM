import cdflib
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import numpy as np
import glob
from pathlib import Path
import geopandas
import time

path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Data/Dayside/Multiple/IPD/20180519_IPD.cdf'
LP_dir = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Data/Dayside/Multiple/LP')
EFI_dir = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Data/Dayside/Multiple/EFI')
IPD_dir = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Data/Dayside/Multiple/IPD')
ACC_dir = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Data/Dayside/Multiple/ACC')

#For electron density, temperature, and surface potential 
# Home/Level1b/Latest_Baseline/EFIx_LP/

def joinDatasets():

    def openEFI(dire):
        cdf_array = []
        cdf_files = dire.glob('*.cdf')
        for f in cdf_files:
            cdf = cdflib.CDF(f) #asign to cdf object

            utc = cdf.varget("Timestamp") #select variables of interest
            lat = cdf.varget("Latitude")
            lon = cdf.varget("Longitude")
            alt = cdf.varget("Radius")
            mlt = cdf.varget("MLT")
            # This has value 0 (midnight) in the anti-sunward direction, 12 (noon) 
            # in the sunward direction and 6 (dawn) and 
            # 18 (dusk) perpendicular to the sunward/anti-sunward line.
            Ti = cdf.varget("Ti_meas_drift")
            TiM = cdf.varget("Ti_model_drift")

            #place in dataframe
            cdf_df = pd.DataFrame({'datetime':utc, 'mlt':mlt, 'lat':lat, 'long':lon, "Ti":Ti, "TiM":TiM})
            cdf_array.append(cdf_df)

        return pd.concat(cdf_array) #concat enables multiple .cdf files to be to one df

    def openIPD(dire):
        
        cdf_array = []
        cdf_files = dire.glob('*.cdf')
        for f in cdf_files:
            cdf = cdflib.CDF(f) #asign to cdf object

            utc = cdf.varget("Timestamp") #select variables of interest
            Te = cdf.varget("Te")
            Ne = cdf.varget("Ne")
            ROD = cdf.varget("ROD") #Rate of change of density in cm^-3/s
            reg = cdf.varget("Ionosphere_region_flag")

            #place in dataframe
            cdf_df = pd.DataFrame({'datetime':utc, 'Te':Te, "Ne":Ne, "rod":ROD, "reg":reg})
            cdf_array.append(cdf_df)

        return pd.concat(cdf_array) #concat enables multiple .cdf files to be to one df

    def openLP(dire):
        cdf_array = []
        cdf_files = dire.glob('*.cdf')
        for f in cdf_files:
            cdf = cdflib.CDF(f) #asign to cdf object

            utc = cdf.varget("Timestamp")
            alt = cdf.varget("Radius")
            Vs = cdf.varget("Vs")

            cdf_df = pd.DataFrame({"datetime":utc, "alt":alt, "pot":Vs})
            cdf_array.append(cdf_df)
            

        return pd.concat(cdf_array) 

    def openACC(dire):
        cdf_xyz = []
        cdf_time = []
        cdf_files = dire.glob('*.cdf')
        for f in cdf_files:
            cdf = cdflib.CDF(f) #asign to cdf object

            utc = cdf.varget("time")
            a_cal = cdf.varget("a_cal") #calibrated linear acceleration (X, Y, Z)
            a_stp = cdf.varget("a_stp") #step corrections for linear accelerations

            #time
            cdf_df = pd.DataFrame({"datetime":utc})
            cdf_time.append(cdf_df)

            #xyz for exploding
            cdf_df = pd.DataFrame({"cal":[a_cal]})
            cdf_xyz.append(cdf_df)

            
        return pd.concat(cdf_xyz), pd.concat(cdf_time)

    
    #Call different instrument data
    EFI_data = openEFI(EFI_dir)
    IPD_data = openIPD(IPD_dir)
    LP_data = openLP(LP_dir)
    ACC_xyz, ACC_time = openACC(ACC_dir)

    
    #ACC_data = ACC_data.explode('cal')
    ACC_xyz = ACC_xyz.apply(pd.Series.explode).values.tolist()
    
    #ACC_time = ACC_time.values.tolist()

    ACC_list = [item[0] for item in ACC_xyz]
    ACC_list = [item[0] for item in ACC_list]
    #print(ACC_list[0])

    
    ACC_x = pd.DataFrame(ACC_list,columns=['accel'])
    ACC_time = ACC_time.reset_index().drop(columns=['index'])

    #print(ACC_x)
    #print(ACC_time)

    #boolean = ACC_time['datetime'].duplicated().any()

    #print(boolean)

    
    ACC_data = pd.concat([ACC_time, ACC_x], axis=1)
    print(ACC_data)

    joined_data = ACC_data

    #csv_output_pathfile = csv_output_path + "/acc-check.csv" # -4 removes '.pkts' or '.dat'
    #csv_output_path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Instrument Data/Analysis/Sept-21/data'
    #ACC_xyz.to_csv(csv_output_pathfile, index = False, header = True)

    #normalise cadence
    LP_data = LP_data.iloc[::2]
    EFI_data = EFI_data.iloc[::2]

    #merge dataframes
    #joined_data = EFI_data.merge(IPD_data, on = 'datetime').merge(LP_data, on ='datetime') #WORKING DO NOT EDIT 16-09
    #joined_data = EFI_data.merge(IPD_data, on = 'datetime').merge(ACC_data, on ='datetime')
    #joined_data = joined_data.dropna()

    # /// filter df ////
    #joined_data = joined_data.loc[joined_data['reg'] < 2]
    #joined_data = joined_data.loc[joined_data['reg'] == 1]  #0 equator, 1 mid-lat, 2 auroral oval, 3 polar cap
    #joined_data = joined_data[~joined_data['mlt'].between(6,18)]
    #joined_data = joined_data[(joined_data.mlt != 6) & (joined_data.mlt != 18)] 
    #joined_data = joined_data[joined_data['lat'].between(-10,10)]
    #joined_data = joined_data[joined_data['long'].between(0,1)]
    #joined_data = joined_data[joined_data['Ti'].between(600,2000)]
    #joined_data = joined_data[joined_data['Te'].between(600,5000)]
    #joined_data = joined_data[::60]

    #print(joined_data)
    '''
    
    # //// transform df ////
    joined_data['alt'] = (joined_data['alt'] / 1000) - 6371 #remove earth radius and refine decimal places
    joined_data['Ne'] = joined_data['Ne'] * 1e6
    joined_data['rod'] = joined_data['rod'] * 1e6
    joined_data["mlt"] = joined_data["mlt"].astype(int)'''


    def convert2Datetime(utc):
        #https://pypi.org/project/cdflib/
        utc = cdflib.epochs.CDFepoch.to_datetime(utc)
        return utc

    # /// Convert from epoch to datetime, then split into two columns ///
    joined_data['datetime'] = joined_data['datetime'].apply(convert2Datetime).str[0].astype(str)
    temp_df = joined_data["datetime"].str.split(" ", n = 1, expand = True)
    joined_data["date"] = temp_df [0]
    joined_data["utc"] = temp_df [1]
    joined_data = joined_data.reset_index().drop(columns=['index'])

    print(joined_data)


    #Select date
    #joined_data = joined_data.loc[joined_data['date'] == '2018-05-19']

    def select_hours(df):
        
        import datetime as dt
        sort_by = 'utc'
        df['hr'] = pd.to_datetime(df[sort_by]).dt.hour
        df.drop_duplicates(subset=['date', 'hr'], keep='first', inplace=True)
        df.drop('hr', axis=1, inplace=True)
        return df
    
    def daynight(x):
        if 6 <= x <= 18:
            return 'day'
        else:
            return 'night'

    def densityDefintion(x):
        if x > 1e11:
            return 'most dense'
        elif 6e10 <= x <= 1e11:
            return 'dense'
        else:
            return 'least dense' 

    def daynightExtra(df):
        if 7 <= df <= 17:
            return 'day'
        elif 5 <= df <= 7:
            return 'dawn'
        elif 17 <= df <= 19:
            return 'dusk'
        else:
            return 'night'
    '''
    joined_data['hemi'] = joined_data['mlt'].apply(daynight)
    joined_data['den'] = joined_data['Ne'].apply(densityDefintion)

    joined_data = joined_data[['date','utc','mlt','hemi','lat','long','alt','Ne','den','Te','Ti','pot','reg']] #re-order dataframe

    #Data reduction techniques
    #joined_data = joined_data.groupby(['date','mlt'], as_index=False)['Ne','Te','Ti'].mean() #Select single MLT per date
    #joined_data = select_hours(joined_data) #Select one hour per day

    #Check label balance
    print (joined_data['den'].value_counts())

    csv_output_path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Instrument Data/Analysis/Sept-21/data'
    csv_output_pathfile = csv_output_path + "/IPD-Dayside-Cleaned.csv" # -4 removes '.pkts' or '.dat'
    joined_to_csv = joined_data.to_csv(csv_output_pathfile, index = False, header = True)

    #print (joined_data)
    return joined_data'''

joined_data = joinDatasets()
#print(joined_data)


def plotSWARM():

    plot_data = joinDatasets()
    print(plot_data)
    #print(plot_data.dtypes)

    #Normalise
    #min_max = preprocessing.MinMaxScaler()
    #norm = min_max.fit_transform()


    #plot_data ['Ne'] = (plot_data['Ne'] - plot_data['Ne'].min()) / (plot_data['Ne'].max() - plot_data['Ne'].min())
    #plot_data ['Te'] = (plot_data['Te'] - plot_data['Te'].min()) / (plot_data['Te'].max() - plot_data['Te'].min())
    #plot_data ['Ti'] = (plot_data['Ti'] - plot_data['Ti'].min()) / (plot_data['Ti'].max() - plot_data['Ti'].min())

    print(plot_data)
    

    #filter
    #plot_data = plot_data.iloc[::1000]
    #print(plot_data)

    #sns.jointplot(data = plot_data, x='Ne', y = 'rod', kind = 'hist')
    
    df = pd.melt(plot_data, id_vars=['mlt','date'], value_vars = ['Ne'])
    #print(df)

    #sns.lineplot(data = plot_data, x = 'mlt', y ='Te', hue = 'date')

    #sns.jointplot(data = plot_data, x = "Ne", y = "Te", hue = 'mlt')

    #sns.boxplot(data=df, x="mlt", y="value", hue = 'variable')
    #sns.despine(offset=10, trim=True)

    #hue, palette = 'variable', 'rocket'
    #sns.lineplot(data = df, x = 'mlt', y = 'value', hue = hue, palette = palette)


    
    g = sns.FacetGrid(df, hue="value",
                  subplot_kws=dict(projection='polar'), height=4.5,
                  sharex=False, sharey=False, despine=False)

    g.map(sns.scatterplot,"mlt", "value")
    #g.map(sns.scatterplot,"utc", "value")

    #plt.gca().set_theta_direction(-1)
    #plt.gca().set_theta_zero_location("N")

    #plt.set_yticklabels([])
    #plt.set_theta_zero_location('N')

    #g.set_theta_direction(-1)

    g.set(xticklabels=[])

    #plt.yscale('log')

    plt.tight_layout()
    plt.show()

