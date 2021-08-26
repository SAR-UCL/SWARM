import cdflib
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import numpy as np
import glob
from pathlib import Path


#path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Data/Dayside/20180519/Level1b/20180519_LP.cdf'
directory = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Data/Dayside/Testing')

def openCDF(dire):
    cdf_list = []
    cdf_files = dire.glob('*.cdf')
    for f in cdf_files:
        cdf = cdflib.CDF(f)
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
        cdf_list.append(LP_data)

    merged_data = pd.concat(cdf_list)
    return merged_data

#merge_LP = pd.concat(LP_data, ignore_index=True)
#print(merge_LP)


'''Working'''
#cdf = cdflib.CDF(path)
#Check which variables are in the df
#cdf_variables = cdf.cdf_info()
#cdf_variables = cdf_variables['zVariables']
#print(cdf_variables)

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


getLP()