import cdflib
import pandas as pd
import glob
from pathlib import Path
#import time

IBI_dir = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/in-flight data/IBI/March-19')
LP_dir = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/in-flight data/LP/March-19')
EFI_dir = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/in-flight data/EFI/March-19')
path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Non-Flight Data/Analysis/Nov-21/data/March-19/'

#file_name = 'LP-data_20211103.h5'

IBI_output = path + 'IBI-data_March-19.h5'
LP_output = path + 'LP-data_March-19.h5'
EFI_output = path + 'EFI-data_March-19.h5'
joined_output = path + 'joined-data_20211104.h5'

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
            #bub_flag = cdf.varget("Flags_Bubble")
            #mag_flag = cdf.varget("Flags_F")

            #place in dataframe
            cdf_df = pd.DataFrame({'datetime':utc,'lat':lat, 'long':lon,'b_ind':bub_ind, 'b_prob':bub_prob})
            cdf_array.append(cdf_df)
            ibi_data = pd.concat(cdf_array)

            #Filters
            #ibi_data = ibi_data.loc[ibi_data['b_ind'] == 1] #1 = Bubble 
            #ibi_data = ibi_data.loc[ibi_data['bub_flag'] == 2] #1 = Confirmed, 2 = Unconfirmed
            #ibi_data = ibi_data.loc[ibi_data['b_prob'] > 0]
            #ibi_data = ibi_data[::30] #30 second cadence to match SOAR
            #ibi_data = ibi_data.drop(columns=['bub_flag','mag_flag']) #reduces DF size

    except RuntimeError:
        raise Exception('Problems extracting IBI files')
    
    def convert2Datetime(utc):
        utc = cdflib.epochs.CDFepoch.to_datetime(utc)
        return utc
    
    ibi_data['datetime'] = ibi_data['datetime'].apply(convert2Datetime).str[0].astype(str)

    #Export
    ibi_data.to_hdf(IBI_output, key = 'ibi_data', mode = 'w')
    print ('IBI file exported.')
    return ibi_data

def openLP(dire):

    cdf_array = []
    cdf_files = dire.glob('**/*.cdf')

    print ("Extracting LP data...")
    try:
        for f in cdf_files:
            cdf = cdflib.CDF(f) #asign to cdf object

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
                "pot_Te":Te_flag, "pot_Ne":ne_flag,"pot_f":Vs_flag}) #flags
            cdf_array.append(cdf_df)
            

            lp_data = pd.concat(cdf_array)
            #lp_data = lp_data[::2]
            lp_data = lp_data.loc[lp_data['pot_Te'] == 20]
            lp_data = lp_data.loc[lp_data['pot_Ne'] == 20]
            lp_data = lp_data.loc[lp_data['pot_f'] == 20]


            lp_data = lp_data.drop(columns=['pot_Te','pot_Ne','pot_f']) #reduces DF size

    except RuntimeError:
        raise Exception('Problems extracting EFI data')

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
                cdf = cdflib.CDF(f) #asign to cdf object

                utc = cdf.varget("Timestamp") #select variables of interest
                #lat = cdf.varget("Latitude")
                #lon = cdf.varget("Longitude")
                #alt = cdf.varget("Radius")
                mlt = cdf.varget("MLT")
                # This has value 0 (midnight) in the anti-sunward direction, 12 (noon) 
                # in the sunward direction and 6 (dawn) and 
                # 18 (dusk) perpendicular to the sunward/anti-sunward line.
                Tn = cdf.varget("Tn_msis")
                Ti = cdf.varget("Ti_meas_drift")
                TiM = cdf.varget("Ti_model_drift")

                #flag 
                Ti_flag = cdf.varget("Flag_ti_meas")

                #place in dataframe
                cdf_df = pd.DataFrame({'datetime':utc, 'mlt':mlt, "Ti":Ti, "Ti_f":Ti_flag})
                cdf_array.append(cdf_df)

                efi_data = pd.concat(cdf_array)
                efi_data = efi_data[::2]
                efi_data = efi_data.loc[efi_data['Ti_f'] == 1]
                efi_data = efi_data.drop(columns=['Ti_f']) #reduces DF size

                efi_data.to_hdf(EFI_output, key = 'efi_data')

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
    
    joined_cdf = read_IBI.merge(read_LP, on = 'datetime').merge(read_EFI, on = 'datetime')

    def splitDatetime(df):
        temp_df = df["datetime"].str.split(" ", n = 1, expand = True)
        df["date"] = temp_df [0]
        df["utc"] = temp_df [1]
        df = df.reset_index().drop(columns=['datetime','index'])
        df = df[['date','utc','mlt','lat','long','alt','b_ind','b_prob','Ne','Ti','pot','Te']]
        return df

    joined_cdf = splitDatetime(joined_cdf)
    print(joined_cdf)

mergeCDF(IBI_output, LP_output, EFI_output)

#print('data successful extracted \n',LP_data)
'''
hdf_IBI = path + 'IBI-data_20211104.h5'
read_IPB = pd.read_hdf(hdf_IBI)

read_LP = pd.read_hdf(LP_output)
read_EFI = pd.read_hdf(EFI_output)

#joined_data = read_LP.merge(read_EFI, on = 'datetime').merge(read_IPB, on='datetime')
joined_data = read_LP.merge(read_EFI, on = 'datetime')
'''
def removeDatetime(df):
    temp_df = df["datetime"].str.split(" ", n = 1, expand = True)
    df["date"] = temp_df [0]
    df["utc"] = temp_df [1]
    df = df.reset_index().drop(columns=['datetime','index'])

    return df

#joined_data = removeDatetime(joined_data)
#joined_data = joined_data.drop_duplicates(subset=['lat'])
#joined_data = joined_data.drop_duplicates(subset=['Te'])
#joined_data.to_hdf(joined_output, key = 'joined_data')
#print(joined_data)


#print(read_LP)

def removeDatetime(df):
    temp_df = df["datetime"].str.split(" ", n = 1, expand = True)
    df["date"] = temp_df [0]
    df["utc"] = temp_df [1]
    #df["utc"] = df['utc'].astype(str).str.slice(stop =-3)

    df = df.reset_index().drop(columns=['datetime','index'])
    #df = df[['date','utc','lat','long','b_ind','b_prob','bub_flag','mag_flag']]

    return df

#joined_data = removeDatetime(read_LP)
#print(clean_hdf)'''

'''
df = clean_hdf.sort_values(by=['date'])
for col in df:
  print(df[col].unique())
  print(len(df[col].unique()))'''

#read_EFI = pd.read_hdf(EFI_output)

#print(read_IPB)
#print(read_IPB, read_LP, read_EFI)

#joined_data = read_LP.merge(read_IPB, on = 'datetime').merge(read_EFI, on ='datetime')
#joined_data = read_IPB.merge(read_EFI, on = 'datetime')


#print(joined_data)


def convert2Datetime(utc):
    utc = cdflib.epochs.CDFepoch.to_datetime(utc)
    return utc

#read_EFI['datetime'] = read_EFI['datetime'].apply(convert2Datetime).str[0].astype(str)
#print(read_EFI)

#joined_data['datetime'] = joined_data['datetime'].apply(convert2Datetime).str[0].astype(str)
#print(joined_data)