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
#pd.set_option('display.max_rows', None) #or 10 or None

dir_suffix = 'A'

IBI_dir = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/in-flight data/IBI/two_sat/C')
LP_dir = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/in-flight data/LP/two_sat/C')
EFI_dir = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/in-flight data/EFI/two_sat/C')
MAG_dir = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/in-flight data/MAG/'+dir_suffix)
path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Non-Flight Data/Analysis/Mar-22/data/two_sat/'

#Output names
IBI_output = path + 'IBI-data_'+dir_suffix+'.csv'
LP_output = path + 'LP-data_'+dir_suffix+'.csv'
EFI_output = path + 'EFI-data_'+dir_suffix+'.csv'
#MAG_output = path + 'MAG-data_'+dir_suffix+'.csv'
EPB_output = path +'/EPB_counts/'+ 'EPB-count-IBI_'+dir_suffix+'.csv'

today =  str(date.today())
joined_output = path + dir_suffix + '-data-'+ today +'.csv'

class extraction():

    def convert2Datetime(self,utc):
        utc = cdflib.epochs.CDFepoch.to_datetime(utc)
        return utc

    def openIBI(self, dire):

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
        

        ibi_data['datetime'] = ibi_data['datetime'].apply(self.convert2Datetime).str[0].astype(str)
        ibi_data = ibi_data.reset_index().drop(columns=['index'])

        #Export
        print('Exporting dataframe...')
        ibi_data.to_csv(IBI_output, index=False, header = True)
        print ('IBI data exported.')
        return ibi_data

    def openLP(self, dire):

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

                lp_data = pd.concat(cdf_array)


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

                #lp_data = calcROC(lp_data)

                def pass_count(df):
                    ml = []
                    start = 0
                    for i in range(len(df.index)):
                            if i % 2700 == 0:
                                    start +=1
                            else:
                                    pass
                            ml.append(start)
                    return ml
                counter = pass_count(lp_data)
                lp_data['p_num'] = counter

                def flags_drop_cols(df):
                    #Remove flags
                    #https://earth.esa.int/eogateway/documents/20142/37627/swarm-level
                    #-1b-product-definition-specification.pdf/12995649-fbcb-6ae2-5302
                    # -2269fecf5a08
                        
                    df = df.loc[df['LP_f'] != 7]
                    df = df.loc[((df['Ne_f'] != 31) &
                            (df['Ne_f'] != 40 ))]
                    df = df.loc[( (df['Te_f'] != 31) & 
                            (df['Te_f'] != 40) & (df['Te_f'] != 41) )]
                    df = df.loc[((df['pot_f'] != 31) &
                            (df['pot_f'] != 32) & (df['pot_f'] != 41))]
                    df = df.drop(columns=['Ne_f','Ne_f','pot_f','Ne_c',
                            'Te_c','pot_c'])

                    return df

                lp_data = flags_drop_cols(lp_data)
                
        except RuntimeError:
            raise Exception('Problems extracting LP data')

        lp_data['datetime'] = lp_data['datetime'].apply(self.convert2Datetime).str[0].astype(str)
        lp_data = lp_data.reset_index().drop(columns=['index'])

        #Export 
        lp_data.to_csv(LP_output, index=False, header = True)
        print ('LP data exported.')
        return lp_data

    def openEFI(self, dire):
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
                    Te = cdf.varget("Te_adj_LP")

                    #flag 
                    Ti_flag = cdf.varget("Flag_ti_meas")

                    #place in dataframe
                    cdf_df = pd.DataFrame({'mlt':mlt, 'datetime':utc,
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
                    #efi_data = efi_data[~efi_data['mlt'].between(6,18)]
                    #efi_data = efi_data.loc[efi_data['Ti_f'] == 1]
                    #efi_data = efi_data.drop(columns=['Ti_f','Ti_c'])

            except RuntimeError:
                raise Exception('Problems extracting EFI data')

            
            efi_data['datetime'] = efi_data['datetime'].apply(self.convert2Datetime).str[0].astype(str)
            efi_data["datetime"] = efi_data['datetime'].str.slice(stop =-4)
                  
            #Export
            efi_data.to_csv(EFI_output, index=False, header = True)
            print ('EFI data exported.')

            return efi_data

    def openMAG(self, dire):
        #Get sat ID

        cdf_scal_array = []
        cdf_vec_array = []
        cdf_files = dire.glob('**/*.cdf')

        print ("Extracting MAG data...")

        try:
            for f in cdf_files:
                cdf = cdflib.CDF(f)
                sat_id = str(f)
                sat_id = sat_id[-61:-60]

                utc = cdf.varget("Timestamp")

                f = cdf.varget("F") #b-field intensity nT
                f_err = cdf.varget("F_error")
                b = cdf.varget("B") #Magnetic field vector, NEC frame 

                #extract scalars
                cdf_scal = pd.DataFrame({'datetime':utc, 'f':f,
                        "f_err":f_err})
                cdf_scal_array.append(cdf_scal)
                mag_scal_data = pd.concat(cdf_scal_array)
                mag_scal_data = mag_scal_data.reset_index().drop(columns=['index'])

                #extract vectors
                cdf_vec = pd.DataFrame({"b":[b]})
                cdf_vec_array.append(cdf_vec)
                mag_vec_data = pd.concat(cdf_vec_array)
                mag_vec_data = mag_vec_data.apply(pd.Series.explode).values.tolist()
                vfm_list = [item[0] for item in mag_vec_data] #access list 1
                mag_vec_data = pd.DataFrame(vfm_list, columns = ['nec_x','nec_y','nec_z'])

                #merge scalars and vectors
                mag_data = pd.concat([mag_scal_data, mag_vec_data], axis =1)

        except RuntimeError:
            raise Exception('Problems extracting MAG data')

        #perform filtering
        #mag_data['f'] = mag_data['f'] - 1000

        mag_data['datetime'] = mag_data['datetime'].apply(self.convert2Datetime).str[0].astype(str)
        mag_data["datetime"] = mag_data['datetime'].str.slice(stop =-4)

        #Export
        mag_data.to_csv(MAG_output, index=False, header = True)
        print ('MAG data exported.')

        return mag_data

    def mergeCDF(self, IBI, LP, EFI):

        print('Loading instruments...')
        #Load cdf's
        read_IBI = pd.read_csv(IBI)
        read_LP = pd.read_csv(LP)
        read_EFI = pd.read_csv(EFI)
        #read_MAG = pd.read_csv(MAG)
        #print(read_IBI, read_LP, read_EFI, read_MAG)
        
        try:
            print ('Joining dataframes...')
            
            df = read_LP.merge(read_EFI, on = ['s_id', 'datetime']).merge(read_IBI, on= 
                    ['datetime', 's_id'])
                    #.merge(read_MAG, on = ['datetime'])
            
            print ('Splitting datetime...')
            def splitDatetime(df):
                #Splits datetime into date & utc, then reorders the df
                temp_df = df["datetime"].str.split(" ", n = 1, expand = True)
                df["date"] = temp_df [0]
                df["utc"] = temp_df [1]
                df = df.reset_index().drop(columns=['datetime','index'])
                return df
            
            df = splitDatetime(df)

            df = df.sort_values(by=['s_id','date','utc'], 
                    ascending = True)

            df = df[['date','utc','mlt','lat','long','alt','s_id', 'p_num',
                    'b_ind','b_prob','Ne','Ne_std','Ti','Ti_std','pot','pot_std',
                    'Te','Te_std']]
                    #,'f','nec_x','nec_y','nec_z']]

            df = df.reset_index().drop(columns=['index'])
            #df = df.drop_duplicates(subset=['nec_z']) 

            #print('Joined dataframe\n',df)'''
        
        except RuntimeError:
            raise Exception('Problems joining dataframes')

        #Export to csv
        df.to_csv(joined_output, index=False, header = True)
        print('Joined dataframes exported')
    
extract = extraction()
#ibi_data = extract.openIBI(IBI_dir)
#lp_data = extract.openLP(LP_dir)
#efi_data = extract.openEFI(EFI_dir)
#mag_data = extract.openMAG(MAG_dir)
merged_data = extract.mergeCDF(IBI_output, LP_output, EFI_output)
#print(merged_data)

#print('Loading data...')
#df = pd.read_csv(joined_output)
#print(df)

def heatmap(df):

    import numpy as np

    #df = df[df['p_num'] == 1]
    #df = df[df['b_ind'] == 1]
    #df = df.describe()
    
    print('Heatmap prep...')
    #df = df[df['date']== '2015-02-01']
    df = df[df['b_ind']!= -1]
    #df = df[df['b_ind'] == 0]
    df = df[df['lat'].between(-35,35)]

    #epb_count = df.groupby(['date','p_num'])['b_ind'].count()
    df = df.groupby(['date','p_num'], as_index=False)['b_ind'].sum()
    
    def count_epb(x):
        if x > 1:
            return 1
        else:
            return 0

    temp_df = df["date"].str.split("-", n = 2, expand = True)
    df["year"] = temp_df [0]
    df["month"] = temp_df [1]
    df["day"] = temp_df [2]
    #df = df[::60]
    df = df.reset_index().drop(columns=['index'])
    df = df.sort_values(by=['date','p_num'], ascending=[True,True])

    df['epb'] = df['b_ind'].apply(count_epb)
    #df = df[df['b_ind'] > 500].reset_index().drop(columns=['index'])

    df.to_csv(EPB_output, index=False, header = True)
    print('EPB Exported.')

    print(df)    
    

    pivot_data = df.pivot_table(values="epb",index="day",columns="month", 
            aggfunc=np.sum, dropna = False)
    print(pivot_data)

    import seaborn as sns 
    from matplotlib import pyplot as plt
    from matplotlib.colors import LogNorm
    
    plt.figure(figsize = (3,6.6))
    sns.heatmap(pivot_data, annot=True, linewidths=0.1, vmin=0,
             fmt=".0f", cmap="YlGnBu")

    plt.title('Number of EPB events in 2014 \n (IBI Classifier)', 
            fontsize=10.5)
    plt.xlabel('Month')
    plt.ylabel('Day')
    plt.yticks(rotation = 0)

    plt.tight_layout()
    plt.show()

#heatmap(df)

