import cdflib
import pandas as pd
import glob
from pathlib import Path
import os
from datetime import date

dir_suffix = 'test'

IBI_dir = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/in-flight data/IBI/'+dir_suffix)
LP_dir = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/in-flight data/LP/'+dir_suffix)
EFI_dir = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/in-flight data/EFI/'+dir_suffix)
#path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Non-Flight Data/Analysis/Jan-22/data/test/'
path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/in-flight data/LP/test/'

#Output names
IBI_output = path + 'IBI-data_'+dir_suffix+'.h5'
LP_output = path + 'LP-data_'+dir_suffix+'.h5'
EFI_output = path + 'EFI-data_'+dir_suffix+'.h5'

today =  str(date.today())
joined_output = path + 'decadal-data-'+ today +'.csv'

class extractCDF():

    def convert2Datetime(self, utc):
        utc = cdflib.epochs.CDFepoch.to_datetime(utc)
        return utc

    def pass_count(self, df):
        ml = []
        start = 0
        for i in range(len(df.index)):
                if i % 2700 == 0:
                        start +=1
                else:
                        pass
                ml.append(start)
        return ml

    def extractLP(self, dire):

        def get_vars():
            #cdf_array=[]
            cdf_files = dire.glob('**/*.cdf')

            #for f in cdf_files:
            #cdf = cdflib.CDF(f)

            cdf = cdflib.CDF(cdf_files)

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

            return lp_data

        def calc_mini_ROC(df):
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

            lp_data = calcROC(lp_data)

        #Internal functions
        lp_data = get_vars() #get variables
        lp_data = calc_mini_ROC(lp_data) #caclulate rates of change

        #class-wide function
        #get time
        lp_data['datetime'] = lp_data['datetime'].apply(self.convert2Datetime).str[0].astype(str)
        counter = self.pass_count(lp_data)
        lp_data['p_num'] = counter
    
        return lp_data


# extract = extractCDF()

# cdf_files = LP_dir.glob('*/*.cdf')
# for f in cdf_files:
#      cdf = cdflib.CDF(f)
#      LP_data = extract.extractLP(LP_dir)
#      print(LP_data)

#LP_data = extract.extractLP(LP_dir)
#print(LP_data)

#dire = LP_dir

class extraction():

    def convert2Datetime(self,utc):
        utc = cdflib.epochs.CDFepoch.to_datetime(utc)
        return utc
    
    def pass_count(self,df):
        ml = []
        start = 0
        for i in range(len(df.index)):
                if i % 2700 == 0:
                        start +=1
                else:
                        pass
                ml.append(start)
        return ml

    def IBI_data(self, cdf, f):
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

        cdf_df = cdf_df[cdf_df['lat'].between(-30,30)] #Nightime only

        #class functions
        counter = self.pass_count(cdf_df)
        cdf_df['p_num'] = counter
        #cdf_df['datetime'] = cdf_df['datetime'].apply(self.convert2Datetime).str[0].astype(str)

        return cdf_df

    def LP_data(self,cdf,f):

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

        #class functions
        #counter = self.pass_count(cdf_df)
        #cdf_df['p_num'] = counter
        #cdf_df['datetime'] = cdf_df['datetime'].apply(self.convert2Datetime).str[0].astype(str)

        return cdf_df

    def get_data_2(self, ibi_dir,lp_dir):
    #def get_data(self,ibi_dir,instrument):

        ibi_arr = []
        ibi_files = ibi_dir.glob('**/*.cdf')
        for f in ibi_files:
            cdf_ibi = cdflib.CDF(f)
            ibi_arr.append(self.IBI_data(cdf_ibi,f))

            lp_arr = []
            lp_files = lp_dir.glob('**/*.cdf')
            for j in lp_files:
                cdf_lp = cdflib.CDF(j)
                lp_arr.append(self.LP_data(cdf_lp,j))
            cdf_data_2 = pd.concat(lp_arr)
            
            #return cdf_array_2
            #LP_data = get_vars(cdf)
            #ibi_arr.append(self.IBI_data(cdf,f))
            #ibi_arr.append(instrument(cdf,f))
            #ibi_arr.append(self.IBI_data(cdf,f))
        cdf_data = pd.concat(ibi_arr)

        merged = cdf_data_2.merge(cdf_data, on = ['datetime','s_id'])
        #merged = pd.concat(merged)

        #merged['datetime'] = merged['datetime'].apply(self.convert2Datetime).str[0].astype(str)

        return merged

    def get_data(self,dire,instrument):

        cdf_array = []
        ibi_files = dire.glob('**/*.cdf')
        for f in cdf_files:
            cdf = cdflib.CDF(f)
            cdf_array.append(instrument(cdf,f))
            cdf_data = pd.concat(cdf_array)
        return cdf_data

extract = extraction()
multi_data = extract.get_data_2(IBI_dir, LP_dir)
print(multi_data)

#lp_data = extract.get_data(LP_dir)
#ibi_data = extract.get_data(IBI_dir, extract.IBI_data)
#lp_data = extract.get_data(LP_dir, extract.LP_data)

#print(ibi_data)
#print(lp_data)

#merged_instruments = ibi_data.merge(lp_data, on = ['datetime','s_id'])
#print(merged_instruments)


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

            lp_data = calcROC(lp_data)

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

            #lp_data = flags_drop_cols(lp_data)
            

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

#LP_data = openLP(LP_dir)
#print(LP_data)