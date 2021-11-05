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
import zipfile

IBI_dir = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/in-flight data/IBI/April-16')
path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Non-Flight Data/Analysis/Nov-21/data/'
file_name = 'IBI-data_20211104.h5'
hdf_output = path + file_name

def openIBI(dire):

    cdf_array = []
    cdf_files = dire.glob('**/*.cdf')
    
    print ("Extracting IBI files...")
    try: 
        for f in cdf_files:
            cdf = cdflib.CDF(f) #asign to cdf object

            #header
            utc = cdf.varget("Timestamp")
            lat = cdf.varget("Latitude")
            lon = cdf.varget("Longitude")

            #sciencer1
            bub_ind = cdf.varget("Bubble_Index")
            bub_prob = cdf.varget("Bubble_Probability")

            #flags
            bub_flag = cdf.varget("Flags_Bubble")
            mag_flag = cdf.varget("Flags_F")


            #place in dataframe
            cdf_df = pd.DataFrame({'datetime':utc,'lat':lat, 'long':lon,'b_ind':bub_ind, 'b_prob':bub_prob,
                'bub_flag':bub_flag, 'mag_flag':mag_flag})
            cdf_array.append(cdf_df)
            ibi_data = pd.concat(cdf_array)

            #Filters
            #ibi_data = ibi_data.loc[ibi_data['b_ind'] == 1] #1 = Bubble 
            #ibi_data = ibi_data.loc[ibi_data['bub_flag'] == 2] #1 = Confirmed, 2 = Unconfirmed
            #ibi_data = ibi_data.loc[ibi_data['b_prob'] > 0]

            #ibi_data = ibi_data[::30] #30 second cadence to match SOAR
            
            ibi_data = ibi_data.drop(columns=['bub_flag','mag_flag']) #reduces DF size

    except RuntimeError:
        raise Exception('Problems extracting IBI files')

    
    def convert2Datetime(utc):
        utc = cdflib.epochs.CDFepoch.to_datetime(utc)
        return utc
    
    ibi_data['datetime'] = ibi_data['datetime'].apply(convert2Datetime).str[0].astype(str)

    #Export
    ibi_data.to_hdf(hdf_output, key = 'ibi_data')
    print ('IBI file exported.')
    return ibi_data

IBI_data = openIBI(IBI_dir)
print('IBI data successful extracted \n',IBI_data)

#read_hdf = pd.read_hdf(hdf_output, key = ""
#print(read_hdf)

#clean_hdf = read_hdf.reset_index().drop(['index'], axis = 1)
#clean_hdf = clean_hdf.loc[clean_hdf['b_prob'] >= 0.95]
#clean_hdf = clean_hdf.loc[clean_hdf['b_prob'].between(0.9, 0.94)]
#oined_data = joined_data[joined_data['Te'].between(600,5000)]

def convert2Datetime(utc):
    utc = cdflib.epochs.CDFepoch.to_datetime(utc)
    return utc

#clean_hdf['datetime'] = clean_hdf['datetime'].apply(convert2Datetime).str[0].astype(str)

#print(clean_hdf)

#clean_hdf.to_hdf(hdf_output, key = 'clean_hdf')

def removeDatetime(df):
    temp_df = df["datetime"].str.split(" ", n = 1, expand = True)
    df["date"] = temp_df [0]
    df["utc"] = temp_df [1]
    #df["utc"] = df['utc'].astype(str).str.slice(stop =-3)

    df = df.reset_index().drop(columns=['datetime','index'])
    df = df[['date','utc','lat','long','b_ind','b_prob','bub_flag','mag_flag']]

    return df

#clean_hdf = removeDatetime(clean_hdf)
'''
df = clean_hdf.sort_values(by=['date'])
for col in df:
  print(df[col].unique())
  print(len(df[col].unique()))'''

'''
print(clean_hdf)
sns.lineplot(data = clean_hdf, x = 'date', y = 'b_prob')
#pivot = clean_hdf.pivot("lat", "long", "b_prob")
#print(pivot)

#sns.jointplot(data=clean_hdf, x="long", y="lat", hue="date")
#sns.boxplot(data = clean_hdf, x="date", y="b_prob")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()'''

'''
#Loading and exporting
path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Instrument Data/Analysis/Sept-21/data/'
file_name = r'/IBI-data.csv'
load_csv = csv_path + file_name
load_csv = pd.read_csv(load_csv)
#load_csv = load_csv.loc[load_csv['bub_flag'] == 1] #bubble flag 1 = confirmed bubble
print(load_csv)


#figs, axs = plt.subplots(ncols=1, nrows=1, figsize=(9.5,6.5), sharex=False, sharey=True) #3.5 for single, #5.5 for double
#axs = axs.flatten()

load_csv.plot(kind="scatter", x="long", y="lat", alpha=0.4,
c="b_prob", cmap=plt.get_cmap("jet"), colorbar=True)

plt.legend()
plt.show()'''