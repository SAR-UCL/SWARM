import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import numpy as np


def convertIPD():

    path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Data/Dayside/20180519/Level2/20180519_IPD.txt'
    csv_output = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Data/Dayside/20180519/Level2/'

    #38
    read_swarm = pd.read_csv(path, delim_whitespace=True, skiprows=38,  dtype={"Latitude":"string","Longitude":"string","RODI10s":"string","RODI20s": "string"})

    clean_swarm = read_swarm[['Record','Timestamp','Radius','Latitude','Longitude','Ne','Te','Ionosphere_region_flag']].reset_index().drop(columns=['index'])
    clean_swarm ['Altitude'] = (clean_swarm['Radius'] / 1000) - 6371
    clean_swarm = clean_swarm.drop(columns=['Radius'])
    clean_swarm = clean_swarm[['Record','Timestamp','Altitude','Latitude','Longitude','Ionosphere_region_flag','Ne','Te']]

    clean_swarm = clean_swarm.rename(columns={"Record": "Date","Timestamp":"UTC","Ionosphere_region_flag":"Io Region"}, errors="raise")

    clean_swarm["Date"] = pd.to_datetime(clean_swarm["Date"]).dt.date
    clean_swarm["UTC"] = pd.to_datetime(clean_swarm["UTC"]).dt.time


    clean_swarm['Latitude'] = clean_swarm.apply(lambda x: x['Latitude'][:5], axis = 1)
    clean_swarm['Longitude'] = clean_swarm.apply(lambda x: x['Longitude'][:5], axis = 1)
    clean_swarm['Latitude'] = clean_swarm['Latitude'].astype(float)
    clean_swarm['Longitude'] = clean_swarm['Longitude'].astype(float)  

    #csv_output_pathfile = csv_output + '20180528_IPD.csv'
    #clean_swarm.to_csv(csv_output_pathfile, index = False, header = True)

    print(clean_swarm)

#convertIPD()

def convertEFI():
    path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Data/Dayside/20180529/Level2/20180529_EFI.txt'
    csv_output = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Data/Dayside/20180529/Level2/'

    #38
    read_swarm = pd.read_csv(path, delim_whitespace=True, skiprows=64,  dtype={"Latitude":"string","Longitude":"string","RODI10s":"string","RODI20s": "string"})

    
    clean_swarm = read_swarm[['Record','Timestamp','Height','Latitude','Longitude','Te_adj_LP','Ti_meas_drift','Ti_model_drift']].reset_index().drop(columns=['index'])
    clean_swarm ['Altitude'] = clean_swarm['Height'] / 1000 
    clean_swarm = clean_swarm.drop(columns=['Height'])
    clean_swarm = clean_swarm.rename(columns={"Record": "Date","Timestamp":"UTC","Te_adj_LP":"Te","Ti_meas_drift":"Ti","Ti_model_drift":"TiM"}, errors="raise")
    
    clean_swarm = clean_swarm[['Date','UTC','Altitude','Latitude','Longitude','Te','Ti','TiM']]

    #clean_swarm = clean_swarm.rename(columns={"Record": "Date","Timestamp":"UTC","Ionosphere_region_flag":"Io Region"}, errors="raise")

    clean_swarm["Date"] = pd.to_datetime(clean_swarm["Date"]).dt.date
    clean_swarm["UTC"] = pd.to_datetime(clean_swarm["UTC"]).dt.time


    clean_swarm['Latitude'] = clean_swarm.apply(lambda x: x['Latitude'][:5], axis = 1)
    clean_swarm['Longitude'] = clean_swarm.apply(lambda x: x['Longitude'][:5], axis = 1)
    clean_swarm['Latitude'] = clean_swarm['Latitude'].astype(float)
    clean_swarm['Longitude'] = clean_swarm['Longitude'].astype(float)  

    csv_output_pathfile = csv_output + '20180529_EFI.csv'
    clean_swarm.to_csv(csv_output_pathfile, index = False, header = True)

    print(clean_swarm)

convertEFI()
#dtypes = clean_swarm.dtyper
#s
#print('Data Types:\n', dtypes)

