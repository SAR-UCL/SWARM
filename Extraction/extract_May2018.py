#. Created by Sachin A. Reddy @ MSSL, UCL
# October 2021

#This script will open raw .cdf files corresponding to different SWARM projects. It will then
#Concatinate them together in a single dataframe. Multiple filters and cleaners are applied
#such as removing flagged data, standardising the candence and creating new features

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

path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Data/Dayside/Multiple/IPD/20180519_IPD.cdf'
LP_dir = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Data/Dayside/Multiple/LP')
EFI_dir = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Data/Dayside/Multiple/EFI')
IPD_dir = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Data/Dayside/Multiple/IPD')
ACC_dir = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Data/Dayside/Multiple/ACC')
MAG_dir = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Data/Dayside/Multiple/MAG')
TII_dir = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Data/nov-21/TII/demo')

#For electron density, temperature, and surface potential 
# Home/Level1b/Latest_Baseline/EFIx_LP/

def joinDatasets():

    #The series of 'OpenX' functions open different level products
    #It selects the required information, removes flagged data
    #and returns a dataframe

    def openEFI(dire):
        cdf_array = []
        cdf_files = dire.glob('*.cdf')
        for f in cdf_files:
            cdf = cdflib.CDF(f) #asign to cdf object

            utc = cdf.varget("Timestamp") #select variables of interest
            lat = cdf.varget("Latitude")
            lon = cdf.varget("Longitude")
            alt = cdf.varget("Radius")
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
            cdf_df = pd.DataFrame({'datetime':utc, 'mlt':mlt, 'lat':lat, 'long':lon, "Ti":Ti, "Ti_f":Ti_flag})
            cdf_array.append(cdf_df)

            efi_data = pd.concat(cdf_array)
            efi_data = efi_data.iloc[::4]
            efi_data = efi_data.loc[efi_data['Ti_f'] == 1]
            efi_data = efi_data.drop(columns=['Ti_f']) #reduces DF size


        return efi_data #concat enables multiple .cdf files to be to one df

    def openIPD(dire):
        
        cdf_array = []
        cdf_files = dire.glob('*.cdf')
        for f in cdf_files:
            cdf = cdflib.CDF(f) #asign to cdf object

            utc = cdf.varget("Timestamp") #select variables of interest
            Te = cdf.varget("Te")
            
            Ne = cdf.varget("Ne")
            Ne10_delta = cdf.varget("delta_Ne10s") #Ne10 indicates density fluctuations smaller than 75km cm^-3
            Ne_f = cdf.varget("Foreground_Ne")
            Ne_b = cdf.varget("Background_Ne")

            ROD = cdf.varget("ROD") #Rate of change of density in cm^-3/s
            ROD10 = cdf.varget("RODI10s") #STD Dev of ROD over 10 seconds #cm^-3/s
            ROD20 = cdf.varget("RODI20s") #STD Dev of ROD over 20 seconds #cm^-3/s

            IPIR = cdf.varget("IPIR_index") #Index for plasma fluctuations and irregularities. 0-3 low, 4-5 medium, > 6 high
            reg = cdf.varget("Ionosphere_region_flag")
            bubble = cdf.varget("IBI_flag")

            #flags
            Ne_flag = cdf.varget("Ne_quality_flag")

            #place in dataframe
            #cdf_df = pd.DataFrame({'datetime':utc, 'Te':Te, "Ne":Ne, "Ne_fore":Ne_f, "Ne_back":Ne_b, "Ne_f":Ne_flag, "rod":ROD, 
            #        "rod10":ROD10, "rod20":ROD20, "Ne10":Ne10_delta, "reg":reg,"IPIR":IPIR})
            cdf_df = pd.DataFrame({'datetime':utc, 'Te':Te, "Ne":Ne, "Ne_f":Ne_flag, "rod":ROD, 
                    "reg":reg,"IPIR":IPIR, "bubble":bubble})
            cdf_array.append(cdf_df)

            IPD_data = pd.concat(cdf_array)
            IPD_data = IPD_data.iloc[::2] #reduce cadency to 1hz

            IPD_data = IPD_data.loc[IPD_data['Ne_f'] == 20000]
            #IPD_data = IPD_data.loc[IPD_data['IPIR'] < 4]
            IPD_data = IPD_data.drop(columns=['Ne_f']) #reduces DF size


        return IPD_data #concat enables multiple .cdf files to be to one df

    def openLP(dire):

        cdf_array = []
        cdf_files = dire.glob('*.cdf')
        for f in cdf_files:
            cdf = cdflib.CDF(f) #asign to cdf object

            utc = cdf.varget("Timestamp")
            alt = cdf.varget("Radius")
            Vs = cdf.varget("Vs")
            Vs_flag = cdf.varget("Flags_Vs")

            cdf_df = pd.DataFrame({"datetime":utc, "alt":alt, "pot":Vs,"pot_f":Vs_flag})
            cdf_array.append(cdf_df)

            lp_data = pd.concat(cdf_array)
            lp_data = lp_data.iloc[::2]
            lp_data = lp_data.loc[lp_data['pot_f'] == 20]

            lp_data = lp_data.drop(columns=['pot_f']) #reduces DF size

            
            
        return lp_data 
    
    def openTII(dire):

        cdf_array = []
        cdf_files = dire.glob('*.cdf')
        for f in cdf_files:
            cdf = cdflib.CDF(f) #asign to cdf object   


            #Scalars
            utc = cdf.varget("Timestamp")
            al_h = cdf.varget("Vixh")
            al_v = cdf.varget("Vixv")
            xt_h = cdf.varget("Viy")
            xt_v = cdf.varget("Viz")

            #Flags
            al_h_err = cdf.varget("Vixh_error")
            al_v_err = cdf.varget("Vixv_error")
            xt_h_err = cdf.varget("Viy_error")
            xt_v_err = cdf.varget("Viz_error")


            cdf_df = pd.DataFrame({"Timestamp":utc,"alh":al_h,"alv":al_v, "xth":xt_h,"xtv":xt_v,
                "alh_e":al_h_err,"alv_e":al_v_err, "xth_e":xt_h_err,"xtv_e":xt_v_err})
            cdf_array.append(cdf_df)

            tii_data = pd.concat(cdf_array)

            #tii_data = tii_data.loc[tii_data['alh'].between(-300,300)]
            #tii_data = tii_data.loc[tii_data['alv'].between(-8000,8000)]
            #tii_data = tii_data.loc[tii_data['xth'].between(-8000,8000)]
            #tii_data = tii_data.loc[tii_data['xtv'].between(-8000,8000)]


            def convert2Datetime(utc):
                #https://pypi.org/project/cdflib/
                utc = cdflib.epochs.CDFepoch.to_datetime(utc)
                return utc

            tii_data['datetime'] = tii_data['Timestamp'].apply(convert2Datetime).str[0].astype(str)
            tii_data["datetime"] = tii_data['datetime'].astype(str).str.slice(stop =-7)

            def removeDatetime(df):
                temp_df = df["datetime"].str.split(" ", n = 1, expand = True)
                df["date"] = temp_df [0]
                df["utc"] = temp_df [1]
                df = df.reset_index().drop(columns=['index'])

                return df

            tii_data = removeDatetime(tii_data)
    
            tii_data = tii_data.loc[tii_data['date'] == '2021-07-01'] 
            #date = tii_data['date'].iloc[0]
            date = 'hi'

            #tii_data['alh_p'] = tii_data['alh'] 

            tii_data = tii_data[::120] # cad is 2hz, 120 reduces to every minute
            tii_data = tii_data[:60:] # cad is 2hz, 120 reduces to every minute
            #tii_data = tii_data[::2]

            figs, axs = plt.subplots(ncols=1, nrows=4, figsize=(11.5,7), dpi=90, sharex=True) #3.5 for single, #5.5 for double
            axs = axs.flatten()

            sns.scatterplot(data = tii_data, ax = axs[0], x = "utc", y = "alh" )
            sns.lineplot(data = tii_data, ax = axs[0], x = "utc", y = "alh" )

            sns.scatterplot(data = tii_data, ax = axs[1], x = "utc", y = "alv" )
            sns.lineplot(data = tii_data, ax = axs[1], x = "utc", y = "alv" )

            sns.scatterplot(data = tii_data, ax = axs[2], x = "utc", y = "xth" )
            sns.lineplot(data = tii_data, ax = axs[2], x = "utc", y = "xth" )

            sns.scatterplot(data = tii_data, ax = axs[3], x = "utc", y = "xtv" )
            sns.lineplot(data = tii_data, ax = axs[3], x = "utc", y = "xtv" )


            axs[0].set_title(f'Along & X-track ion drifts, horizontal & vertical \n {date}', size = 10.5)
            axs[3].tick_params(axis='x',labelrotation=90)
            axs[3].set_xlabel(' ')
            
            plt.tight_layout()
            plt.show()

            

            #Single Plot
            '''
            plt.figure(figsize=(11.5,4.5), dpi=90)
            sns.scatterplot(data = tii_data, x = "utc", y = "alh" )
            sns.lineplot(data = tii_data, x = "utc", y = "alh" )
            plt.title('ESA valid min max \n -8000 to 8000 m/s \n cad 1 min', fontsize = 10.5)
            plt.ylabel('Along Track Horizontal Drift [m/s]')
            plt.xlabel('Time')
            plt.xticks(rotation=90)'''
            '''
            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False)''' 


            '''
            f = cdf.varget("F")
            f_err = cdf.varget("Flags_F")
            lat = cdf.varget("Latitude")
            lon = cdf.varget("Longitude")

            #Vectors
            vfm = cdf.varget("B_VFM")
            nec = cdf.varget("B_NEC")
            v_err = cdf.varget("Flags_B")

            #scalars df
            cdf_scalar = pd.DataFrame({"datetime":utc,"F":f,"F_err":f_err,"V_err":v_err})
            cdf_sca.append(cdf_scalar)
            cdf_sca_c = pd.concat(cdf_sca)
            cdf_sca_c = cdf_sca_c.reset_index().drop(columns=['index'])

            #vectors df
            cdf_scalar = pd.DataFrame({"vfm":[vfm],'nec':[nec]})
            cdf_vec.append(cdf_scalar)
            cdf_vec_c = pd.concat(cdf_vec)

            MAG_vec = cdf_vec_c.apply(pd.Series.explode).values.tolist()
            vfm_list = [item[0] for item in MAG_vec] #access list 1
            vfm_df = pd.DataFrame(vfm_list, columns = ['vfm_x','vfm_y','vfm_z']) #x is positive northward, y is positive eastward
            #nec_list = [item[1] for item in MAG_vec] #access list 1
            #nec_df = pd.DataFrame(nec_list, columns = ['nec_x','nec_y','nec_z'])

            #concatinate scalars and vectors
            mag_data = pd.concat([cdf_sca_c, vfm_df], axis =1)

            mag_data = mag_data.loc[mag_data['F_err'] == 1]
            mag_data = mag_data.loc[mag_data['V_err'] == 0]
            mag_data = mag_data.drop(columns=['F_err','V_err']) #reduces DF size'''


        return tii_data

    def openLP(dire):

        cdf_array = []
        cdf_files = dire.glob('*.cdf')
        for f in cdf_files:
            cdf = cdflib.CDF(f) #asign to cdf object

            utc = cdf.varget("Timestamp")
            alt = cdf.varget("Radius")
            Vs = cdf.varget("Vs")
            Vs_flag = cdf.varget("Flags_Vs")

            cdf_df = pd.DataFrame({"datetime":utc, "alt":alt, "pot":Vs,"pot_f":Vs_flag})
            cdf_array.append(cdf_df)

            lp_data = pd.concat(cdf_array)
            lp_data = lp_data.iloc[::2]
            lp_data = lp_data.loc[lp_data['pot_f'] == 20]

            lp_data = lp_data.drop(columns=['pot_f']) #reduces DF size

            
        return lp_data 

    '''
    #Call different instrument data
    EFI_data = openEFI(EFI_dir)
    IPD_data = openIPD(IPD_dir)
    LP_data = openLP(LP_dir)
    MAG_data = openMAG(MAG_dir)'''

    #Testing
    TII_data = openTII(TII_dir)

    return TII_data
    
    '''
    #merge dataframes
    joined_data = EFI_data.merge(IPD_data, on = 'datetime').merge(LP_data, on ='datetime')
    joined_data = joined_data[joined_data['Ti'].between(600,2000)]
    joined_data = joined_data[joined_data['Te'].between(600,5000)]'''

    #joined_data = joined_data.dropna()
    #print(joined_data)
    def processCDF(joined_data):

        def convert2Datetime(utc):
            #https://pypi.org/project/cdflib/
            utc = cdflib.epochs.CDFepoch.to_datetime(utc)
            return utc

        joined_data['datetime'] = joined_data['datetime'].apply(convert2Datetime).str[0].astype(str)
        joined_data["datetime"] = joined_data['datetime'].astype(str).str.slice(stop =-4) #MAG data does not have miliseconds, so this normalises the set before merging
        MAG_data['datetime'] = MAG_data['datetime'].apply(convert2Datetime).str[0].astype(str)

        joined_data = joined_data.merge(MAG_data, on = 'datetime')
        joined_data = joined_data.dropna()
        #joined_data = joined_data[::60] #set cadency (in seconds)
        #print(joined_data)

        #print(joined_data)
        
        def removeDatetime(df):
            temp_df = df["datetime"].str.split(" ", n = 1, expand = True)
            df["date"] = temp_df [0]
            df["utc"] = temp_df [1]
            df = df.reset_index().drop(columns=['index'])

            return df

        joined_data = removeDatetime(joined_data)

        # /// filter df ////
        #joined_data = joined_data.loc[joined_data['reg'] < 2]
        #joined_data = joined_data.loc[joined_data['reg'] == 1]  #0 equator, 1 mid-lat, 2 auroral oval, 3 polar cap
        #joined_data = joined_data[~joined_data['mlt'].between(6,18)]
        #joined_data = joined_data[(joined_data.mlt != 6) & (joined_data.mlt != 18)] 
        #joined_data = joined_data[joined_data['lat'].between(-10,10)]
        #joined_data = joined_data[joined_data['long'].between(0,1)]
        joined_data = joined_data[joined_data['Ti'].between(600,2000)]
        joined_data = joined_data[joined_data['Te'].between(600,5000)]
        #joined_data = joined_data[::60] #set cadency (in seconds)
        
        # //// transform df ////
        joined_data['alt'] = (joined_data['alt'] / 1000) - 6371 #remove earth radius and refine decimal places
        joined_data['Ne'] = joined_data['Ne'] * 1e6
        joined_data['rod'] = joined_data['rod'] * 1e6
        joined_data["mlt"] = joined_data["mlt"].astype(int)

        
        #Select date
        #joined_data = joined_data.loc[joined_data['date'] == '2018-05-19']

        def select_hours(df):
            
            import datetime as dt
            sort_by = 'utc'
            df['hr'] = pd.to_datetime(df[sort_by]).dt.hour
            df.drop_duplicates(subset=['date', 'hr'], keep='first', inplace=True)
            df.drop('hr', axis=1, inplace=True)
            return df
        
        def daynight(x):
            if 6 <= x <= 18:
                return 'day'
            else:
                return 'night'   
        
        def midnightNoon(x):
            if x == 0:
                return 'midnight'
            elif x == 12:
                return 'noon'
            else:
                return 'other'   

        def densityDefintion(x):
            if x > 1e11:
                return 'most dense'
            elif 6e10 <= x <= 1e11:
                return 'dense'
            else:
                return 'least dense' 
        
        def tempDefintion(x):
            if x > 1300:
                return 'hot'
            elif 850 <= x <= 1300:
                return 'thermal'
            else:
                return 'cool'

        def daynightExtra(df):
            if 7 <= df <= 17:
                return 'day'
            elif 5 <= df <= 7:
                return 'dawn'
            elif 17 <= df <= 19:
                return 'dusk'
            else:
                return 'night'
        
        joined_data['hemi'] = joined_data['mlt'].apply(daynight)
        #joined_data = joined_data.loc[joined_data['bubble'] == 1] 
        #joined_data['mid-noon'] = joined_data['mlt'].apply(midnightNoon)
        #joined_data['den'] = joined_data['Ne'].apply(densityDefintion)
        #joined_data['Ti_cat'] = joined_data['Ti'].apply(tempDefintion)

        #joined_data = joined_data[['date','utc','mlt','hemi','lat','long','alt','reg','Ne','rod','Te','Ti','Tn','pot','b_field_int']] #re-order dataframe
        joined_data = joined_data.drop(columns=['datetime'])

        #Data reduction techniques
        #joined_data = joined_data.groupby(['date','mlt'], as_index=False)['Ne','Te','Ti'].mean() #Select single MLT per date
        #joined_data = select_hours(joined_data) #Select one hour per day

        #Check label balance
        #print (joined_data['den'].value_counts())

        csv_output_path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Non-Flight Data/Analysis/Oct-21/data'
        csv_output_pathfile = csv_output_path + "/may18-cleaned.csv" # -4 removes '.pkts' or '.dat'
        joined_data.to_csv(csv_output_pathfile, index = False, header = True)

        #print (joined_data)
        return joined_data

    #processedCDF = processCDF(joined_data)
    #return processedCDF

joined_data = joinDatasets()
print('Whole function applied \n', joined_data)

def plotSWARM():

    plot_data = joinDatasets()
    print(plot_data)
    #print(plot_data.dtypes)

    path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Instrument Data/Analysis/Sept-21/data/'
    file_name = r'IPD-Dayside-Cleaned.csv'
    load_csv = path + file_name
    load_csv = pd.read_csv(load_csv)

    plot_data = load_csv

    #Normalise
    #min_max = preprocessing.MinMaxScaler()
    #norm = min_max.fit_transform()


    #plot_data ['Ne'] = (plot_data['Ne'] - plot_data['Ne'].min()) / (plot_data['Ne'].max() - plot_data['Ne'].min())
    #plot_data ['Te'] = (plot_data['Te'] - plot_data['Te'].min()) / (plot_data['Te'].max() - plot_data['Te'].min())
    #plot_data ['Ti'] = (plot_data['Ti'] - plot_data['Ti'].min()) / (plot_data['Ti'].max() - plot_data['Ti'].min())

    print(plot_data)
    

    #filter
    #plot_data = plot_data.iloc[::1000]
    #print(plot_data)

    #sns.jointplot(data = plot_data, x='Ne', y = 'rod', kind = 'hist')
    
    df = pd.melt(plot_data, id_vars=['mlt','date'], value_vars = ['Ne'])
    #print(df)

    #sns.lineplot(data = plot_data, x = 'mlt', y ='Te', hue = 'date')

    #sns.jointplot(data = plot_data, x = "Ne", y = "Te", hue = 'mlt')

    #sns.boxplot(data=df, x="mlt", y="value", hue = 'variable')
    #sns.despine(offset=10, trim=True)

    #hue, palette = 'variable', 'rocket'
    #sns.lineplot(data = df, x = 'mlt', y = 'value', hue = hue, palette = palette)


    
    g = sns.FacetGrid(df, hue="value",
                  subplot_kws=dict(projection='polar'), height=4.5,
                  sharex=False, sharey=False, despine=False)

    g.map(sns.scatterplot,"mlt", "value")
    #g.map(sns.scatterplot,"utc", "value")

    #plt.gca().set_theta_direction(-1)
    #plt.gca().set_theta_zero_location("N")

    #plt.set_yticklabels([])
    #plt.set_theta_zero_location('N')

    #g.set_theta_direction(-1)

    g.set(xticklabels=[])

    #plt.yscale('log')

    plt.tight_layout()
    plt.show()


    #return cdf_vec_c, cdf_sca_c   
    #return pd.concat(cdf_vec), pd.concat(cdf_sca)

'''
mag_data = openMAG(MAG_dir)
#print(mag_data)

#https://geomag.nrcan.gc.ca/mag_fld/comp-en.php
mag_data ['horizontal'] = np.sqrt((mag_data['vfm_x']**2) + (mag_data['vfm_y']**2)) #nT
mag_data ['declination'] = np.tan(mag_data['vfm_y']/ mag_data['vfm_x'])**-1 #degrees
mag_data ['inclination'] = np.tan(mag_data['vfm_z']/ mag_data['horizontal'])**-1 #degress
mag_data = mag_data[::60]
mag_data = mag_data[mag_data['declination'].between(-90,90)]
mag_data = mag_data[mag_data['inclination'].between(-90,90)]
print(mag_data)


figs, axs = plt.subplots(ncols=2, nrows=2, figsize=(8.5,5.5), sharex=False, sharey=True) #3.5 for single, #5.5 for double
axs = axs.flatten()

mag_data.plot(kind="scatter", x="long", y="lat", alpha=1, ax = axs[0],
c="horizontal", cmap=plt.get_cmap("jet"), colorbar=True)

mag_data.plot(kind="scatter", x="long", y="lat", alpha=1, ax = axs[1],
c="F", cmap=plt.get_cmap("jet"), colorbar=True)

den = r'm$^{-3}$'
mag_data.plot(kind="scatter", x="long", y="lat", alpha=0.4, ax = axs[2],
c="declination", cmap=plt.get_cmap("jet"), colorbar=True)

rod = r'm$^{-3}$/s'
mag_data.plot(kind="scatter", x="long", y="lat", alpha=0.4, ax = axs[3],
c="inclination", cmap=plt.get_cmap("jet"), colorbar=True)

plt.tight_layout()
plt.legend()
plt.show()'''