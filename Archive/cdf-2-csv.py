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

#path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Data/Dayside/Multiple/IPD/20180519_IPD.cdf'
LP_dir = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Data/Dayside/Multiple/LP')
EFI_dir = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Data/Dayside/Multiple/EFI')
IPD_dir = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Data/Dayside/Multiple/IPD')

#For electron density, temperature, and surface potential 
# Home/Level1b/Latest_Baseline/EFIx_LP/

def openLP(dire):
    cdf_array = []
    cdf_files = dire.glob('*.cdf')
    for f in cdf_files:
        cdf = cdflib.CDF(f) #asign to cdf object

        utc = cdf.varget("Timestamp") #select variables of interest
        #lat = cdf.varget("Latitude")
        #lon = cdf.varget("Longitude")
        alt = cdf.varget("Radius")
        Te = cdf.varget("Te")
        Ne = cdf.varget("Ne")
        Vs = cdf.varget("Vs")

        #place in dataframe
        #cdf_df = pd.DataFrame({'datetime':utc, 'lat':lat, 'alt':alt, 'long':lon, 'Te':Te, "Ne":Ne, "pot":Vs})
        cdf_df = pd.DataFrame({'datetime':utc, 'alt':alt,  'Te':Te, "Ne":Ne, "pot":Vs})
        cdf_array.append(cdf_df)

    return pd.concat(cdf_array) #concat enables multiple .cdf files to be to one df

def openEFI(dire):
    cdf_array = []
    cdf_files = dire.glob('*.cdf')
    for f in cdf_files:
        cdf = cdflib.CDF(f) #asign to cdf object

        utc = cdf.varget("Timestamp") #select variables of interest
        lat = cdf.varget("Latitude")
        lon = cdf.varget("Longitude")
        mlt = cdf.varget("MLT")
        # This has value 0 (midnight) in the anti-sunward direction, 12 (noon) 
        # in the sunward direction and 6 (dawn) and 
        # 18 (dusk) perpendicular to the sunward/anti-sunward line.
        Ti = cdf.varget("Ti_meas_drift")
        #TiM = cdf.varget("Ti_model_drift")

        #place in dataframe
        cdf_df = pd.DataFrame({'datetime':utc, 'lat':lat, 'long':lon, 'mlt':mlt, "Ti":Ti})
        cdf_array.append(cdf_df)

    return pd.concat(cdf_array) #concat enables multiple .cdf files to be to one df

def openIPD(dire):
    cdf_array = []
    cdf_files = dire.glob('*.cdf')
    for f in cdf_files:
        cdf = cdflib.CDF(f) #asign to cdf object

        utc = cdf.varget("Timestamp") #select variables of interest
        lat = cdf.varget("Latitude")
        lon = cdf.varget("Longitude")
        Te = cdf.varget("Te")
        Ne = cdf.varget("Ne")
        ROD = cdf.varget("RODI10s")
        reg = cdf.varget("Ionosphere_region_flag")

        #place in dataframe
        cdf_df = pd.DataFrame({'datetime':utc, 'lat':lat, 'long':lon, 'Te':Te, "Ne":Ne, "rod":ROD, "reg":reg})
        cdf_array.append(cdf_df)

    return pd.concat(cdf_array) #concat enables multiple .cdf files to be to one df

def joinDatasets():

    LP_data = openLP(LP_dir)
    EFI_data = openEFI(EFI_dir)
    IPD_data = openIPD(IPD_dir)

    def convert2Datetime(utc):
        #https://pypi.org/project/cdflib/
        utc = cdflib.epochs.CDFepoch.to_datetime(utc)
        return utc

    # /// Convert from epoch to datetime, then split into two columns
    #LP_data['datetime'] = LP_data['datetime'].apply(convert2Datetime).str[0].astype(str) #epoch to datetime
    EFI_data['datetime'] = EFI_data['datetime'].apply(convert2Datetime).str[0].astype(str) #epoch to datetime
    IPD_data['datetime'] = IPD_data['datetime'].apply(convert2Datetime).str[0].astype(str) #epoch to datetime

    #LP_data = LP_data.reset_index().drop(columns=['index'])

    joined_data = pd.merge(LP_data, EFI_data, how ="left", left_on=['datetime'], right_on = ['datetime'])

    # /// Filter data ///
    joined_data = joined_data[joined_data['lat'].between(-10,10)]
    joined_data = joined_data[joined_data['long'].between(0,1)]
    joined_data = joined_data[joined_data['Te'].between(1800,3000)]

    # /// Transform data ///
    temp_df = joined_data["datetime"].str.split(" ", n = 1, expand = True)
    joined_data["date"] = temp_df [0]
    joined_data["utc"] = temp_df [1]
    joined_data = joined_data.reset_index().drop(columns=['index'])

    #joined_data = joined_data.loc[joined_data['date'] == '2018-05-25']
    joined_data['alt'] = (joined_data['alt'] / 1000) - 6371 #remove earth radius and refine decimal places
    joined_data['Ne'] = joined_data['Ne'] * 1e6
    joined_data = joined_data[['date','utc','mlt','lat','long','alt','Te','Ne','Ti']] #re-order

    print(joined_data)


#joinDatasets()

#Needed for new .cdf files to determine var names

#cdf = cdflib.CDF(path)
#cdf_variables = cdf.cdf_info()
#cdf_variables = cdf_variables['zVariables']
#print(cdf_variables)


def getmultiLP():
    
    LP_data = openEFI(EFI_dir)
    #LP_data = openIPD(IPD_dir) #call OpenCDF func 

    # /// Filter data. This goes here to improve performance ////
    #LP_data = LP_data[LP_data['lat'].between(-10,10)]
    #LP_data = LP_data[LP_data['long'].between(0,1)]
    #LP_data = LP_data[LP_data['Te'].between(1800,3000)]

    LP_data = LP_data.iloc[::2] #removes every other row
    
    def convert2Datetime(utc):
        #https://pypi.org/project/cdflib/
        utc = cdflib.epochs.CDFepoch.to_datetime(utc)
        return utc

    # /// Convert from epoch to datetime, then split into two columns
    LP_data['datetime'] = LP_data['datetime'].apply(convert2Datetime).str[0].astype(str) #epoch to datetime
    temp_df = LP_data["datetime"].str.split(" ", n = 1, expand = True)
    LP_data["date"] = temp_df [0]
    LP_data["utc"] = temp_df[1]
    #LP_data["utc"] = pd.to_datetime(LP_data['utc'])
    LP_data = LP_data.reset_index().drop(columns=['index'])
    #LP_data = LP_data.dropna()
    #lower_cad = LP_data[LP_data.utc != '696']

    print(LP_data)
    #print(lower_cad)
    #print(lower_cad.dtypes)


    '''
    # /// Transform data ///
    LP_data = LP_data.loc[LP_data['date'] == '2018-05-25']
    LP_data['alt'] = (LP_data['alt'] / 1000) - 6371 #remove earth radius and refine decimal places
    LP_data['Ne'] = LP_data['Ne'] * 1e6
    LP_data = LP_data[['date','utc','lat','long','alt','Te','Ne','pot']] #re-order

    #print(LP_data)

    #PLOT-TINGS
    plot_LP = LP_data 
    plot_LP['utc'] = plot_LP.apply(lambda x: x['utc'][:5], axis = 1)
    #load_swenix['date'] = load_swenix.apply(lambda x: x['date'][:5], axis = 1)
    #print(plot_LP)
    
    #plt.figure(figsize=(10.5,3.5), dpi=90)
    #sns.lineplot(data = plot_LP, x = 'utc', y = 'pot')

    #Geo Plots
    swarm = geopandas.GeoDataFrame(plot_LP, geometry = geopandas.points_from_xy(plot_LP.long, plot_LP.lat))

    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    ax = world.plot(color='white', edgecolor='black', figsize=(7.5, 3.5))
    #gdf.plot(ax=ax, column = column,# markersize = 20, legend = True, legend_kwds={'shrink': 0.75}, norm = LogNorm())
    #swarm.plot(ax=ax, markersize = 10, legend = True)
    swarm.plot(ax=ax, markersize = 0.01, cmap = 'rainbow')
    
    #plt.yscale('log')

    plt.tight_layout()
    plt.show()'''

getmultiLP()


def getLP():
    '''
    #Extract variables form .cdf
    utc = cdf.varget("Timestamp")
    lat = cdf.varget("Latitude")
    lon = cdf.varget("Longitude")
    alt = cdf.varget("Radius")
    Te = cdf.varget("Te")
    Ne = cdf.varget("Ne")
    Vs = cdf.varget("Vs")

    #place in dataframe
    LP_data = {'datetime':utc, 'lat':lat, 'alt':alt, 'long':lon, 'Te':Te, "Ne":Ne, "pot":Vs}
    LP_data = pd.DataFrame(LP_data)

    #Filter data. This goes here to improve performance'''
    LP_data = openCDF(directory)
    #LP_data = LP_data[LP_data['lat'].between(-89,-87)]
    LP_data = LP_data[LP_data['long'].between(0,1)]
    LP_data = LP_data[LP_data['Te'].between(1000,3000)]


    def convert2Datetime(utc):
        #https://pypi.org/project/cdflib/
        utc = cdflib.epochs.CDFepoch.to_datetime(utc)
        return utc

    def splitDatetime(utc):
        temp_df = LP_data["datetime"].str.split(" ", n = 1, expand = True)
        LP_data["date"] = temp_df [0]
        LP_data["utc"] = temp_df [1]

    #Perform transforms
    
    #Time related
    LP_data['datetime'] = LP_data['datetime'].apply(convert2Datetime).str[0].astype(str) #epoch to datetime
    LP_data['datetime'] = LP_data['datetime'].apply(splitDatetime).drop(columns=['datetime']) #split into date | time

    LP_data = LP_data.reset_index().drop(columns=['index'])

    LP_data['alt'] = (LP_data['alt'] / 1000) - 6371 #remove earth radius and refine decimal places
    LP_data['Ne'] = LP_data['Ne'] * 1e6
    LP_data = LP_data[['date','utc','lat','long','alt','Te','Ne','pot']] #re-order

    #print(LP_data)

    plot_LP = LP_data 

    plot_LP['utc'] = plot_LP.apply(lambda x: x['utc'][:5], axis = 1)
    #load_swenix['date'] = load_swenix.apply(lambda x: x['date'][:5], axis = 1)
    print(plot_LP)

    '''
    plt.figure(figsize=(10.5,3.5), dpi=90)
    sns.lineplot(data = plot_LP, x = 'utc', y = 'pot')
    
    #plt.yscale('log')

    plt.tight_layout()
    plt.show()'''


def permCheck():
    start_time = time.process_time()
    #main()
    print(time.process_time() - start_time, "seconds")

#permCheck()

#getLP()