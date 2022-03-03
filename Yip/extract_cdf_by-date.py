import cdflib
import pandas as pd
import glob
from pathlib import Path
import os
from datetime import date

dir_suffix = 'April-16'

IBI_dir = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/in-flight data/IBI/'+dir_suffix)
LP_dir = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/in-flight data/LP/'+dir_suffix)
EFI_dir = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/in-flight data/EFI/'+dir_suffix)
MAG_dir = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/in-flight data/MAG/'+dir_suffix)
path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Non-Flight Data/Analysis/Feb-22/data/april-mag/'

#Output names
IBI_output = path + 'IBI-data_'+dir_suffix+'.h5'
LP_output = path + 'LP-data_'+dir_suffix+'.h5'
EFI_output = path + 'EFI-data_'+dir_suffix+'.h5'
MAG_output = path + 'MAG-data_'+dir_suffix+'.h5'

today =  str(date.today())
joined_output = path + 'mag-data-'+ today +'.csv'
#joined_output_2 = path + 'test-single-pass-'+ today +'.csv'

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

    def flags_drop_cols(self,df):
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
            #df = df.drop(columns=['Ne_f','Ne_f','pot_f','Ne_c',
            #        'Te_c','pot_c'])

            return df

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

        #cdf_df = cdf_df[cdf_df['lat'].between(-30,30)] #Nightime only

        #class functions
        counter = self.pass_count(cdf_df)
        cdf_df['p_num'] = counter
        #cdf_df['datetime'] = cdf_df['datetime'].apply(self.convert2Datetime).str[0].astype(str)

        #cdf_df['datetime'] = cdf_df['datetime'].apply(self.convert2Datetime).str[0].astype(str)
        #cdf_df["datetime"] = cdf_df['datetime'].str.slice(stop =-4)

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

        def calc_small_ROC(df):
            #Rate of change cm/s or k/s or pot/s
            pc_df = df[['Ne','pot']].pct_change(periods=1) #change in seconds
            pc_df = pc_df.rename(columns = {"Ne":"Ne_c", "pot":"pot_c"}) 
            df = pd.concat([df, pc_df], axis=1)

            #std deviation over change over x seconds
            #How far, on average, the results are from the mean
            std_df = df[['Ne_c','pot_c']].rolling(10).std()
            std_df = std_df.rename(columns = {"Ne_c":"Ne_std_s", "pot_c":"pot_std_s"}) 
            df = pd.concat([df,std_df], axis = 1)

            #df = df.drop(columns=['Ne_c','pot_c'])

            df = df.dropna()

            return df

        def calc_large_ROC(df):
            #Rate of change cm/s or k/s or pot/s
            pc_df = df[['Ne','pot']].pct_change(periods=1) #change in seconds
            pc_df = pc_df.rename(columns = {"Ne":"Ne_c2", "pot":"pot_c2"}) 
            df = pd.concat([df, pc_df], axis=1)

            #std deviation over change over x seconds
            #How far, on average, the results are from the mean
            #std_df = df[['Ne_c2','pot_c2']].rolling(60).std()
            std_df = df[['Ne_c2','pot_c2']].rolling(10, win_type='gaussian').sum(std=4)
            #std_df = df[['Ne_c2','pot_c2']].rolling(10, win_type='kaiser').sum(beta=1)
            std_df = std_df.rename(columns = {"Ne_c2":"Ne_std_l", "pot_c2":"pot_std_l"}) 
            df = pd.concat([df,std_df], axis = 1)

            df = df.dropna()

            return df

        cdf_df = calc_small_ROC(cdf_df)
        cdf_df = calc_large_ROC(cdf_df)

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
            df = df.drop(columns=['Ne_f','Ne_f','pot_f','Ne_c', 'Te_f', 'LP_f',
                    'pot_c','Ne_c2', 'pot_c2'])

            return df
        cdf_df = flags_drop_cols(cdf_df)
        
        cdf_df = cdf_df[cdf_df['Te'].between(0,5000)]

        return cdf_df

    def EFI_data(self,cdf, f):
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
        cdf_df = pd.DataFrame({'mlt':mlt, 'Te':Te,
                "Ti":Ti, "Ti_f":Ti_flag, "s_id":sat_id})

        cdf_df = cdf_df[::2]

        cdf_df = cdf_df[~cdf_df['mlt'].between(6,18)]
        cdf_df = cdf_df[cdf_df['Ti'].between(0,3000)]

        #cdf_df = cdf_df[cdf_df['Ti_f'] == 5]

        #cdf_df['datetime'] = cdf_df['datetime'].apply(self.convert2Datetime).str[0].astype(str)
        #cdf_df["datetime"] = cdf_df['datetime'].str.slice(stop =-4)

        return cdf_df
    
    def MAG_data(self, cdf, f):
        #Get sat ID
        
        sat_id = str(f)
        sat_id = sat_id[-61:-60]

        utc = cdf.varget("Timestamp")

        f = cdf.varget("F") #b-field intensity nT
        f_err = cdf.varget("F_error")
        b = cdf.varget("B") #Magnetic field vector, NEC frame 

        #place in dataframe
        cdf_df = pd.DataFrame({'datetime':utc, 'f':f,
                "f_err":f_err})

        cdf_df['datetime'] = cdf_df['datetime'].apply(self.convert2Datetime).str[0].astype(str)
        cdf_df["datetime"] = cdf_df['datetime'].str.slice(stop =-4)

        #cdf_df = pd.DataFrame({"b":[b]})

        return cdf_df

    '''
    def get_mag_data(self, mag_dir):
        #Electric Field Instrument, l2 product
        mag_arr = []
        mag_files = mag_dir.glob('**/*.cdf')
        for k in mag_files:
                cdf_mag = cdflib.CDF(k)
                mag_arr.append(self.MAG_data(cdf_mag, k))
                mag_data = pd.concat(mag_arr)
        
        return mag_data'''

    def get_instru_data(self, ibi_dir,lp_dir, efi_dir, mag_dir):
        
        #ionospheric bubble index, l2 product
        ibi_arr = []
        ibi_files = ibi_dir.glob('**/*.cdf')
        for i in ibi_files:
            print('Starting extraction...')
            cdf_ibi = cdflib.CDF(i)
            ibi_arr.append(self.IBI_data(cdf_ibi,i))

            #Langmuir Probe, l1 product
            lp_arr = []
            lp_files = lp_dir.glob('**/*.cdf')
            for j in lp_files:
                cdf_lp = cdflib.CDF(j)
                lp_arr.append(self.LP_data(cdf_lp,j))
                
                #Electric Field Instrument, l2 product
                efi_arr = []
                efi_files = efi_dir.glob('**/*.cdf')
                for k in efi_files:
                    cdf_efi = cdflib.CDF(k)
                    efi_arr.append(self.EFI_data(cdf_efi, k))

                    #Magnetometer data, l1 product
                    mag_arr = []
                    mag_files = mag_dir.glob('**/*.cdf')
                    for k in mag_files:
                        cdf_mag = cdflib.CDF(k)
                        mag_arr.append(self.MAG_data(cdf_mag, k))

            mag_data = pd.concat(mag_arr)

            efi_data = pd.concat(efi_arr)
                
            lp_data = pd.concat(lp_arr)
            
            ibi_data = pd.concat(ibi_arr)

            merged = lp_data.merge(efi_data, on = ['s_id', 'Te']).merge(ibi_data, on =['datetime','s_id'])
            #merge = mag_data

            merged['datetime'] = merged['datetime'].apply(self.convert2Datetime).str[0].astype(str)

            merged = merged.merge(mag_data, on = ['datetime'])
            
            def splitDatetime(df):
                temp_df = df["datetime"].str.split(" ", n = 1, expand = True)
                df["date"] = temp_df [0]
                df["utc"] = temp_df [1]
                df = df.reset_index().drop(columns=['datetime','index'])
        
                return df
        
            merged = splitDatetime(merged)
            
            #print(merge)

        #print(merged)
        return merged
        #print(merged)
        #return lp_data
        #return merge
        #return mag_data
        
extract = extraction()
#get_mag = extract.get_mag_data(MAG_dir)
#print(get_mag)

multi_data = extract.get_instru_data(IBI_dir, LP_dir, EFI_dir, MAG_dir)
print(multi_data)
#multi_data.to_csv(joined_output, index=False, header = True)
print('data exported')

#######
#Load csv
#df = pd.read_csv(joined_output)

#df = df[df['b_ind'] == 1]
#df = df[df['p_num'] == 2]
#df = df[df['lat'].between(-15,15)]
#df = df[df['date'] == '2014-01-15']
#df = df.sort_values(by=['utc'], ascending=True)

def gauss_check(x):
    if x <= -0.4 or x>0.4:
        return 1
    else:
        return 0

#df['g_epb'] = df['Ne_std_l'].apply(gauss_check)

def stddev_check(x):
    if x > 0.2:
        return 1
    else:
        return 0

#df['stdev_epb_1'] = df['Ne_std_l'].apply(stddev_check)

def stddev_check_2(x):
    if x > 0.1:
        return 1
    else:
        return 0

#df['stdev_epb_2'] = df['Ne_std_l'].apply(stddev_check_2)

def gauss_stddev(x, y):
        if x or y == 1:
                return 1
        else:
                return 0 

#df['gau-dev'] = df.apply(lambda x: gauss_stddev(x.g_epb, x.stdev_epb_1), axis=1)

#print(df)

#df.to_csv(joined_output_2, index=False, header = True)
#print('data exported')

#from matplotlib import pyplot as plt
#import seaborn as sns
#import numpy as np


def plotNoStdDev(df):
        
        #df = df[df[''] == 32]
        #df = df[df['p_num'] == 32]
        
        figs, axs = plt.subplots(ncols=1, nrows=7, figsize=(10,7), 
        dpi=90, sharex=True) #3.5 for single, #5.5 for double
        axs = axs.flatten()

        x = 'lat'
        #palette_ne, palette_ti, palette_pot = 'Set1', 'Set2', 'tab10'
        #palette_ne, palette_ti, palette_pot = 'flag', 'flag', 'flag'
        hue = 's_id'
        sns.lineplot(ax = axs[0], data = df, x = x, y ='gau-dev', 
                palette = 'bone',hue = hue, legend=False)

        sns.lineplot(ax = axs[1], data = df, x =x, y ='Ne',
                palette = 'flag', hue = hue, legend=False)

        sns.lineplot(ax = axs[2], data = df, x = x, y ='Ne_std_s', 
                #marker = 'o', linestyle='', err_style='bars', 
                palette = 'flag', hue = hue, legend = False)

        sns.lineplot(ax = axs[3], data = df, x = x, y ='Ne_std_l', 
                palette = 'flag', hue = hue, legend = False)

        sns.lineplot(ax = axs[4], data = df, x = x, y ='g_epb', 
                palette = 'flag', hue = hue, legend = False)
        
        sns.lineplot(ax = axs[5], data = df, x = x, y ='stdev_epb_1', 
                palette = 'flag', hue = hue, legend = False)
        
        #ax6 = axs[6].twinx()
        sns.lineplot(ax = axs[6], data = df, x = x, y ='gau-dev', 
                palette = 'flag', hue = hue, legend = False)

        date_s = df['date'].iloc[0]
        date_e = df['date'].iloc[-1]
        utc_s = df['utc'].iloc[0]
        utc_e = df['utc'].iloc[-1]
      
        lat_s = df['lat'].iloc[0]
        lat_e = df['lat'].iloc[-1]

        epb_len = (lat_s - lat_e) * 110
        epb_len = "{:.0f}".format(epb_len)
        
        #print(epb_len)
        #axs[0].set_title(f'Equatorial Plasma Bubble: from {date_s} at {utc_s} to {date_e} at {utc_e}', fontsize = 11)

        epb_check = df['b_ind'].sum()
        if epb_check > 0:
            title = 'Equatorial Plasma Bubble'
        else:
            title = 'Quiet Period'

        sat = 'A'
        pass_num = 'N/A'

        axs[0].set_title(f'{title}: from {date_s} at {utc_s} ' 
                f'to {date_e} at {utc_e}. Spacecraft: {sat}, Pass: '
                f'{pass_num}', fontsize = 11)
        axs[0].set_ylabel('EPB \n (Man)')
        #axs[0].set_ylim(0, 1)
        axs[0].tick_params(bottom = False)
        #axs[0].axhline( y=0.9, ls='-.', c='k')

        axs[1].set_ylabel('Ne')
        axs[1].tick_params(bottom = False)
        axs[1].set_yscale('log')

        #left, bottom, width, height = (1, 0, 14, 7)
        #axs[4].add_patch(Rectangle((left, bottom),width, height, alpha=1, facecolor='none'))

        
        den = r'cm$^{-3}$'
        axs[2].set_ylabel(f'Ne std')
        axs[2].tick_params(bottom = False)
        #axs[3].set_yscale('log')


        axs[3].set_ylabel('Ne gauss')
        axs[3].tick_params(bottom = False)
        #axs[3].set_yscale('log')

        axs[4].set_ylabel('Gauss EPB')
        axs[4].tick_params(bottom = False)
        #axs[4].set_xlabel(' ')


        axs[5].set_xlabel('Stddev EPB')
        #axs[5].set_ylabel('UTC')
        axs[5].tick_params(left = False)

        axs[6].set_xlabel('pot')
        #axs[6].set_ylabel('UTC')
        axs[6].tick_params(left = False)


        #ax6.set_ylabel('MLT')

        n = len(df) // 3.5
        #[l.set_visible(False) for (i,l) in 
        #        enumerate(axs[5].yaxis.get_ticklabels()) if i % n != 0]


        ax = plt.gca()
        ax.invert_xaxis()

        plt.tight_layout()
        plt.show()

#plotNoStdDev(df)