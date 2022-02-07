'''
    This file extracts the data from the raw .cdf files and converts
    them into an .hdf. 
    It also joins multiple instrument / products for ease of analysis

    Created by Sachin A. Reddy

    November 2021.
'''

import cdflib
import pandas as pd
import glob
from pathlib import Path
import os
from datetime import date
import numpy as np

IBI_dir = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/in-flight data/IBI/orbyts')
LP_dir = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/in-flight data/LP/orbyts')
EFI_dir = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/in-flight data/EFI/orbyts')
path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Non-Flight Data/Analysis/Jan-22/data/orbyts/'

#Output names
IBI_output = path + 'IBI-data_decadal.h5'
LP_output = path + 'LP-data_decadal.h5'
EFI_output = path + 'EFI-data_decadal.h5'

today =  str(date.today())
joined_output = path + 'orbyts-data-'+ today +'.csv'


def openIBI(dire):

    cdf_array = []
    cdf_files = dire.glob('**/*.cdf')
    
    print ("Extracting IBI data...")
    try: 
        for f in cdf_files:
            cdf = cdflib.CDF(f) #assign to cdf object

            #Get sat ID
            sat_id = str(f)
            sat_id = sat_id[-61:-60]

            #header
            utc = cdf.varget("Timestamp")
            lat = cdf.varget("Latitude")
            lon = cdf.varget("Longitude")

            #science
            bub_ind = cdf.varget("Bubble_Index")
            bub_prob = cdf.varget("Bubble_Probability")

            #flags
            #bub_flag = cdf.varget("Flags_Bubble")
            #mag_flag = cdf.varget("Flags_F")

            #place in dataframe
            cdf_df = pd.DataFrame({'datetime':utc,'lat':lat, 'long':lon,
                    'b_ind':bub_ind, 'b_prob':bub_prob,
                    's_id':sat_id})
            cdf_array.append(cdf_df)
            ibi_data = pd.concat(cdf_array)

            #Filters & Flags
            #ibi_data = ibi_data.loc[ibi_data['b_ind'] == 1] #1 = Bubble 
            #ibi_data = ibi_data.loc[ibi_data['bub_flag'] == 2] #1 = Confirmed, 2 = Unconfirmed
            #ibi_data = ibi_data.loc[ibi_data['b_prob'] > 0]
            #ibi_data = ibi_data[::30] #30 second cadence to match SOAR
            #ibi_data = ibi_data.drop(columns=['bub_flag','mag_flag']) #reduces DF size

    except RuntimeError:
        raise Exception('Problems extracting IBI data')
    
    def convert2Datetime(utc):
        utc = cdflib.epochs.CDFepoch.to_datetime(utc)
        return utc
    
    ibi_data['datetime'] = ibi_data['datetime'].apply(convert2Datetime).str[0].astype(str)
    ibi_data = ibi_data.reset_index().drop(columns=['index'])

    #Export
    ibi_data.to_hdf(IBI_output, key = 'ibi_data', mode = 'w')
    print ('IBI data exported.')
    return ibi_data

def openLP(dire):

    cdf_array = []
    cdf_files = dire.glob('**/*.cdf')

    print ("Extracting LP data...")
    try:
        for f in cdf_files:
            cdf = cdflib.CDF(f)

            sat_id = str(f)
            sat_id = sat_id[-72:-71]

            utc = cdf.varget("Timestamp")
            alt = cdf.varget("Radius")

            Te = cdf.varget("Te")
            Ne = cdf.varget("Ne")
            Vs = cdf.varget("Vs")
            
            #Flags
            #info https://earth.esa.int/eogateway/documents/20142/37627/swarm-level-1b-plasma-processor-algorithm.pdf
            LP_flag = cdf.varget("Flags_LP")
            Te_flag = cdf.varget("Flags_Te")
            ne_flag = cdf.varget("Flags_Ne")
            Vs_flag = cdf.varget("Flags_Vs")

            cdf_df = pd.DataFrame({"datetime":utc, "alt":alt, "Ne":Ne, "Te":Te, 
                "pot":Vs,"LP_f":LP_flag,"Te_f":Te_flag, "Ne_f":ne_flag,
                "pot_f":Vs_flag,"s_id":sat_id})
            cdf_array.append(cdf_df)

            def calcROC(df):
            
                #Rate of change cm/s or k/s or pot/s
                pc_df = df[['Ne','Te','pot']].pct_change(periods=1) #change in seconds
                pc_df = pc_df.rename(columns = {"Ne":"Ne_c", "Te":"Te_c", "pot":"pot_c"}) 
                df = pd.concat([df, pc_df], axis=1)

                #std deviation over change over x seconds
                #How far, on average, the results are from the mean
                std_df = df[['Ne_c','Te_c','pot_c']].rolling(10).std()
                std_df = std_df.rename(columns = {"Ne_c":"Ne_std", "Te_c":"Te_std", "pot_c":"pot_std"}) 
                df = pd.concat([df,std_df], axis = 1)

                df = df.dropna()

                return df

            lp_data = pd.concat(cdf_array)
            lp_data = calcROC(lp_data)

            #Remove flags
            #https://earth.esa.int/eogateway/documents/20142/37627/swarm-level
            #-1b-product-definition-specification.pdf/12995649-fbcb-6ae2-5302
            # -2269fecf5a08
            
            lp_data = lp_data.loc[lp_data['LP_f'] != 7]
            lp_data = lp_data.loc[((lp_data['Ne_f'] != 31) &
                    (lp_data['Ne_f'] != 40 ))]
            lp_data = lp_data.loc[( (lp_data['Te_f'] != 31) & 
                    (lp_data['Te_f'] != 40) & (lp_data['Te_f'] != 41) )]
            lp_data = lp_data.loc[((lp_data['pot_f'] != 31) &
                    (lp_data['pot_f'] != 32) & (lp_data['pot_f'] != 41))]
            lp_data = lp_data.drop(columns=['Ne_f','Ne_f','pot_f','Ne_c',
                    'Te_c','pot_c'])

    except RuntimeError:
        raise Exception('Problems extracting LP data')

    def convert2Datetime(utc):
        utc = cdflib.epochs.CDFepoch.to_datetime(utc)
        return utc
        
    lp_data['datetime'] = lp_data['datetime'].apply(convert2Datetime).str[0].astype(str)
    lp_data = lp_data.reset_index().drop(columns=['index'])

    #Export 
    lp_data.to_hdf(LP_output, key = 'lp_data', mode = 'w')
    print ('LP data exported.')
    return lp_data

def openEFI(dire):
        cdf_array = []
        cdf_files = dire.glob('**/*.cdf')

        print ("Extracting EFI data...")

        try:
            for f in cdf_files:
                cdf = cdflib.CDF(f)

                #Get sat ID
                sat_id = str(f)
                sat_id = sat_id[-61:-60]

                utc = cdf.varget("Timestamp")
                mlt = cdf.varget("MLT")
                #Tn = cdf.varget("Tn_msis")
                Ti = cdf.varget("Ti_meas_drift")
                #TiM = cdf.varget("Ti_model_drift")

                #flag 
                Ti_flag = cdf.varget("Flag_ti_meas")

                #place in dataframe
                cdf_df = pd.DataFrame({'datetime':utc, 'mlt':mlt, 
                        "Ti":Ti, "Ti_f":Ti_flag, "s_id":sat_id})
                cdf_array.append(cdf_df)

                efi_data = pd.concat(cdf_array)

                def calcROC(df):
                
                    #Rate of change cm/s or k/s or pot/s
                    pc_df = df[['Ti']].pct_change(periods=1) #change in seconds
                    pc_df = pc_df.rename(columns = {"Ti":"Ti_c"}) 
                    df = pd.concat([df, pc_df], axis=1)

                    #std deviation over change over x 
                    #How far, on average, the results are from the mean
                    std_df = df[['Ti_c']].rolling(10).std()
                    std_df = std_df.rename(columns = {"Ti_c":"Ti_std"}) 
                    df = pd.concat([df,std_df], axis = 1)

                    df = df.dropna()

                    return df
        
                efi_data = calcROC(efi_data)
                
                efi_data = efi_data[::2] #reduce to 1hz
                #efi_data = efi_data.loc[efi_data['Ti_f'] == 1]
                #efi_data = efi_data.drop(columns=['Ti_f','Ti_c'])

        except RuntimeError:
            raise Exception('Problems extracting EFI data')

        def convert2Datetime(utc):
            utc = cdflib.epochs.CDFepoch.to_datetime(utc)
            return utc
        
        efi_data['datetime'] = efi_data['datetime'].apply(convert2Datetime).str[0].astype(str)
        efi_data["datetime"] = efi_data['datetime'].str.slice(stop =-4)
        
        #Export
        efi_data.to_hdf(EFI_output, key = 'efi_data', mode = 'w')
        print ('EFI data exported.')
        return efi_data #concat enables multiple .cdf files to be to one df

##Load open functions
#IBI_data = openIBI(IBI_dir)
#EFI_data = openEFI(EFI_dir)
#LP_data = openLP(LP_dir)
#print(LP_data)
#print(IBI_data, EFI_data)

def mergeCDF(IBI, LP, EFI):

    print ('Loading instruments...')
    #Load cdf's
    read_IBI = pd.read_hdf(IBI)
    read_LP = pd.read_hdf(LP)
    read_EFI = pd.read_hdf(EFI)
    #print(read_IBI, read_LP, read_EFI)

    #df = read_LP
    #df = df.sort_values(by=['datetime'], ascending = True)
    #print(df)

    
    try:
        print ('Joining dataframes...')
        #Join the different dataframes
        joined_cdf = read_IBI.merge(read_LP, on = 
                ['datetime','s_id']).merge(read_EFI, on = ['datetime','s_id'])

        #Splits datetime into date & utc, then reorders the df
        def splitDatetime(df):
            temp_df = df["datetime"].str.split(" ", n = 1, expand = True)
            df["date"] = temp_df [0]
            df["utc"] = temp_df [1]
            df = df.reset_index().drop(columns=['datetime','index'])
        
            return df
        
        joined_cdf = splitDatetime(joined_cdf)
        joined_cdf = joined_cdf.sort_values(by=['s_id','date','utc'], 
                ascending = True)
        #joined_cdf = joined_cdf[['date','utc','mlt','lat','long','alt','s_id',
        #        'b_ind','b_prob','Ne','Ne_std','Ti','Ti_std','pot','pot_std',
        #        'Te','Te_std']]

        joined_cdf = joined_cdf[['date','utc','mlt','lat','long','alt','Ne','Te','Ti']]

        temp_df = joined_cdf["date"].str.split("-", n = 2, expand = True)
        joined_cdf["year"] = temp_df [0]
        joined_cdf["month"] = temp_df [1]
        joined_cdf = joined_cdf[::45]
        joined_cdf = joined_cdf.reset_index().drop(columns=['index'])
        joined_cdf['alt'] = (joined_cdf['alt'] / 1000 ) - 6371
        #print('Joined dataframe\n',joined_cdf)
    
    except RuntimeError:
        raise Exception('Problems joining dataframes')

    #print(joined_cdf)

    #joined_cdf.to_hdf(joined_output, key = 'efi_data', mode = 'w')
    joined_cdf.to_csv(joined_output, index=False, header = True)
    print('Joined dataframes exported')

    #return joined_cdf'''

#merged_cdf = mergeCDF(IBI_output, LP_output, EFI_output)

df = pd.read_csv(joined_output)
print(df)
#df = df[::60]

def heatmap():

    pivot_data = df.pivot_table("Ne","month","year", aggfunc=np.mean, dropna = False)

    import seaborn as sns 
    from matplotlib import pyplot as plt
    from matplotlib.colors import LogNorm

    cbar = 'rocket'
    sns.heatmap(pivot_data, annot=False, cbar=cbar, linewidths=0.1, vmin=0, 
            norm = LogNorm(), fmt=".0f")

    plt.show()
        
def daynight(x):
    if 6 > x < 18:
        return 'dayside'
    else:
        return 'nightside'   
    
def latitudes(x):
    if 60 <= x <= 90:
        return 'high-lat'
    elif 30 < x < 60:
        return 'mid-lat'
    elif 10 < x < 30:
        return 'low-lat'
    elif -10 < x < 10:
        return 'equator'
    elif -30 < x < -10:
        return 'low-lat'
    elif -60 < x < -30:
        return 'mid-lat'
    else:
        return 'high-lat'   

def seasons(x):
    if 0 > x < 3:
        return 'winter'
    elif 2 > x < 6:
        return 'spring'
    elif 5 > x < 9:
        return 'summer'
    elif 8 > x < 12:
        return 'autumn'
    else:
        return 'winter'   

df['time'] = df['mlt'].apply(daynight)
df['latitudes'] = df['lat'].apply(latitudes)

df = df.loc[df['Te'] <= 5000]
df = df.loc[df['Te'] > 0]

df = df.loc[df['Ti'] <= 5000]
df = df.loc[df['Ti'] > 0]

#df['seasons'] = df['month'].apply(seasons)

#heatmap()


df.to_csv(joined_output, index=False, header = True)

print(df)