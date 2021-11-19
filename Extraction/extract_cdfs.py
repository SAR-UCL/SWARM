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


IBI_dir = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/in-flight data/IBI/April-16')
LP_dir = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/in-flight data/LP/April-16')
EFI_dir = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/in-flight data/EFI/April-16')
path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Non-Flight Data/Analysis/Nov-21/data/April-16/'

#Output names
IBI_output = path + 'IBI-data_April-16.h5'
LP_output = path + 'LP-data_April-16.h5'
EFI_output = path + 'EFI-data_April-16.h5'

today =  str(date.today())
joined_output = path + 'joined-data-'+ today +'-Apr.h5'

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

            #sciencer1
            bub_ind = cdf.varget("Bubble_Index")
            bub_prob = cdf.varget("Bubble_Probability")

            #flags
            #bub_flag = cdf.varget("Flags_Bubble")
            #mag_flag = cdf.varget("Flags_F")

            #place in dataframe
            cdf_df = pd.DataFrame({'datetime':utc,'lat':lat, 'long':lon,'b_ind':bub_ind, 'b_prob':bub_prob,'s_id':sat_id})
            cdf_array.append(cdf_df)
            ibi_data = pd.concat(cdf_array)

            #Filters
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
            LP_flah = cdf.varget("Flags_LP")
            Te_flag = cdf.varget("Flags_Te")
            ne_flag = cdf.varget("Flags_Ne")
            Vs_flag = cdf.varget("Flags_Vs")

            cdf_df = pd.DataFrame({"datetime":utc, "alt":alt, "Ne":Ne, "Te":Te, "pot":Vs,
                "pot_Te":Te_flag, "pot_Ne":ne_flag,"pot_f":Vs_flag, "s_id":sat_id})
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
            lp_data = lp_data.loc[lp_data['pot_Te'] == 20]
            lp_data = lp_data.loc[lp_data['pot_Ne'] == 20]
            lp_data = lp_data.loc[lp_data['pot_f'] == 20]
            lp_data = lp_data.drop(columns=['pot_Te','pot_Ne','pot_f','Ne_c','Te_c','pot_c'])

    except RuntimeError:
        raise Exception('Problems extracting LP data')

    def convert2Datetime(utc):
        utc = cdflib.epochs.CDFepoch.to_datetime(utc)
        return utc
        
    lp_data['datetime'] = lp_data['datetime'].apply(convert2Datetime).str[0].astype(str)

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
                cdf_df = pd.DataFrame({'datetime':utc, 'mlt':mlt, "Ti":Ti, "Ti_f":Ti_flag, "s_id":sat_id})
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
                efi_data = efi_data.loc[efi_data['Ti_f'] == 1]
                efi_data = efi_data.drop(columns=['Ti_f','Ti_c'])

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

#Load open functions
#IBI_data = openIBI(IBI_dir)
#LP_data = openLP(LP_dir)
#EFI_data = openEFI(EFI_dir)
#print(IBI_data, LP_data, EFI_data)

def mergeCDF(IBI, LP, EFI):

    #Load cdf's
    read_IBI = pd.read_hdf(IBI)
    read_LP = pd.read_hdf(LP)
    read_EFI = pd.read_hdf(EFI)
    #print(read_IBI, read_LP, read_EFI)
    
    try:
        print ('Joining dataframes...')
        #Join the different dataframes
        joined_cdf = read_IBI.merge(read_LP, on = ['datetime','s_id']).merge(read_EFI, on = ['datetime','s_id'])

        #Splits datetime into date & utc, then reorders the df
        def splitDatetime(df):
            temp_df = df["datetime"].str.split(" ", n = 1, expand = True)
            df["date"] = temp_df [0]
            df["utc"] = temp_df [1]
            df = df.reset_index().drop(columns=['datetime','index'])
        
            return df
        
        joined_cdf = splitDatetime(joined_cdf)
        joined_cdf = joined_cdf.sort_values(by=['s_id','date','utc'], ascending = True)
        #print('Joined dataframe\n',joined_cdf)
    
    except RuntimeError:
        raise Exception('Problems joining dataframes')

    try:
        print('Transforming dataframe...')
        
        def calcROC(df):
            
            #Rate of change cm/s or k/s or pot/s
            #pc_df = df[['Ne','Ti','pot','Te']].pct_change(periods=1) #change in seconds
            #pc_df = pc_df.rename(columns = {"Ne":"Ne_c", "Ti":"Ti_c", "pot":"pot_c", "Te":"Te_c"}) 
            #df = pd.concat([df, pc_df], axis=1)

            #std deviation over change over x seconds
            #How far, on average, the results are from the mean
            #std10_df = df[['Ne_c','Ti_c','pot_c']].rolling(5).std()
            #std20_df = df[['Ne','Ti_c','pot_c']].rolling(5).std()  
            #std10_df = std10_df.rename(columns = {"Ne_c":"Ne_std5", "Ti_c":"Ti_std10", "pot_c":"pot_std10"}) 
            #std20_df = std20_df.rename(columns = {"Ne":"Ne_5", "Ti_c":"Ti_std5", "pot_c":"pot_std5"})  
            #df = pd.concat([df,std10_df,std20_df], axis = 1)

            df = df[['date','utc','mlt','lat','long','alt','s_id','b_ind','b_prob',
                    'Ne','Ne_std','Ti','Ti_std','pot','pot_std','Te','Te_std']]
            df = df.dropna()

            return df
        
        print('Transformed dataframe.')

        joined_cdf = calcROC(joined_cdf)
        #print(joined_cdf)

    except RuntimeError:
        raise Exception('Problems transforming dataframe')

    def tempCats(x, y):
        if x > 0.095:
            return 1
        else:
            return 0

    joined_cdf['temp_prob'] = joined_cdf.apply(lambda x: tempCats(x.Ne_std, x.pot_std), axis=1)

    def newEPB(x, y):
        import numpy as np
        if x or y == np.abs(1):
            return 1
        else:
            return 0

    joined_cdf['n_prob'] = joined_cdf.apply(lambda x: newEPB(x.b_ind, x.temp_prob), axis=1)
    #print(joined_cdf)

    joined_cdf.to_hdf(joined_output, key = 'efi_data', mode = 'w')
    print('Joined dataframes exported')

merged_cdf = mergeCDF(IBI_output, LP_output, EFI_output)

def createFeatures(df):
    #file_name = 'joined-data-2021-11-17-Apr.h5'
    #load_hdf = path + file_name
    #print(df)

#createFeatures(merged_cdf)