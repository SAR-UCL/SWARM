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

IBI_dir = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/in-flight data/IBI')
csv_path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Non-Flight Data/Analysis/Nov-21/data'


def openIBI(dire):
    cdf_array = []
    #cdf_open = glob.glob(
    cdf_files = dire.glob('**/*.cdf')
    for f in cdf_files:
        cdf = cdflib.CDF(f) #asign to cdf object

        #header
        utc = cdf.varget("Timestamp") #select variables of interest
        lat = cdf.varget("Latitude")
        lon = cdf.varget("Longitude")
        #loc = cdf.varget("Ionosphere_region_flag")

        #science
        bub_ind = cdf.varget("Bubble_Index")
        bub_prob = cdf.varget("Bubble_Probability")

        #flags
        bub_flag = cdf.varget("Flags_Bubble")
        mag_flag = cdf.varget("Flags_F")

        #place in dataframe
        #cdf_df = pd.DataFrame({'datetime':utc,'lat':lat, 'long':lon, 'loc':loc, 'b_ind':bub_ind})
        cdf_df = pd.DataFrame({'datetime':utc,'lat':lat, 'long':lon,'b_ind':bub_ind, 'b_prob':bub_prob,
            'bub_flag':bub_flag, 'mag_flag':mag_flag})
        cdf_array.append(cdf_df)
        ibi_data = pd.concat(cdf_array)


        ibi_data = ibi_data.loc[ibi_data['b_ind'] == 1]
        #ibi_data = ibi_data[ibi_data['long'].between(-75,45)]
        
        #Filters
        #ibi_data = ibi_data.loc[ibi_data['b_ind'] == 1]
        #ibi_data = ibi_data.loc[ibi_data['bub_flag'] == 2]
        ibi_data = ibi_data.loc[ibi_data['b_prob'] > 0]

        
        csv_output_pathfile = csv_path + "/IBI-data.csv" # -4 removes '.pkts' or '.dat'
        ibi_data.to_csv(csv_output_pathfile, index = False, header = True)


    return ibi_data
    #return efi_data #concat enables multiple .cdf files to be to one df

#IBI_data = openIBI(IBI_dir)
#print(IBI_data)


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
plt.show()