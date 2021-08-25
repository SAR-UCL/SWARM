import cdflib
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import numpy as np

path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Data/Dayside/20180519/Level1b/20180519_LP.cdf'
#path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Data/Dayside/20180519/Level1b/20180519_LPI.cdf'

cdf = cdflib.CDF(path)

cdf_variables = cdf.cdf_info()
cdf_variables = cdf_variables['zVariables']
#print(cdf_variables)

def getLP():

    utc = cdf.varget("Timestamp")
    lat = cdf.varget("Latitude")
    lon = cdf.varget("Longitude")
    alt = cdf.varget("Radius")
    Te = cdf.varget("Te")
    Ne = cdf.varget("Ne")
    Ne = [den * 1e6 for den in Ne]
    Vs = cdf.varget("Vs")

    LP_data = {'datetime':utc, 'lat':lat, 'alt':alt, 'long':lon, 'Te':Te, "Ne":Ne, "Potential":Vs}
    LP_data = pd.DataFrame(LP_data)
    #print(LP_data)

    #LP_data['utc'] = LP_data['utc'].apply(utc)
    LP_data['alt'] = (LP_data['alt'] / 1000) - 6371

    #Filter the different files out
    LP_data = LP_data[LP_data['Te'].between(1000,3000)]
    LP_data = LP_data[LP_data['lat'].between(-89,-87)]

    #split_dataframe = lambda x: pd.Series([i for i in reversed(x.split(' '))])

    def convert2Datetime(utc):
        #https://pypi.org/project/cdflib/
        utc = cdflib.epochs.CDFepoch.to_datetime(utc)
        return utc

    def splitDatetime(utc):
        temp_df = LP_data["datetime"].str.split(" ", n = 1, expand = True)
        LP_data["date"] = temp_df [0]
        LP_data["utc"] = temp_df [1]

    LP_data = LP_data.reset_index().drop(columns=['index']) 
    LP_data['datetime'] = LP_data['datetime'].apply(convert2Datetime).str[0].astype(str)
    LP_data['datetime'] = LP_data['datetime'].apply(splitDatetime)
    LP_data = LP_data.drop(columns=['datetime'])


    LP_df = LP_data[['date','utc','lat','long','alt','Te','Ne','Potential']]

    #print(LP_data)
    print(LP_df)
    #print (LP_data.dtypes)

    #sns.lineplot(data = LP_data, x = 'utc', y = 'Te')
    #plt.show()

getLP()