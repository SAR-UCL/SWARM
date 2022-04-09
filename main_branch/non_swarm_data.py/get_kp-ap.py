from ftplib import FTP
import pandas as pd
import os
import glob
from pathlib import Path
import numpy as np

download_path = Path(r'/Users/sr2/OneDrive - University College ' 
    'London/PhD/Research/Models/kp')

def download_kp(dire):
    os.chdir(dire)

    ktp = FTP('ftp.gfz-potsdam.de')
    ktp.login()
    ktp.cwd('/pub/home/obs/Kp_ap_Ap_SN_F107')

    years = [2014,2015,2016,2017,2018,2018,2019,2020,2021]

    for i in years:
        filename = 'Kp_ap_'+str(i)+'.txt'
        print(f'Downloading data for {i}')
        with open(filename, 'wb' ) as file :
            ktp.retrbinary('RETR %s' % filename, file.write)
            file.close()

    ktp.quit()

#download_kp(download_path)

def transform_kp(dire):

    df_array = []
    kp_files = dire.glob('**/*.txt')

    for i in kp_files:
        kp_dat = pd.read_csv(i, on_bad_lines='skip', sep=" ")

        kp_dat = kp_dat.iloc[18:].reset_index().drop(columns=['level_0','index','ap','index.1','three-hour','interval)','distributes','the','Kp','per'])
        kp_dat = kp_dat.rename(columns={'#':'yy','PURPOSE:':'mm','This':'dd','file':'hr','and':'Kp','(one':'ap1','line':'ap2'})
        kp_dat['date'] = kp_dat['yy'] +'-'+ kp_dat['mm'] +'-'+ kp_dat['dd']
        kp_dat = kp_dat.drop(['yy', 'mm','dd'], axis=1)
        kp_dat['hr']= kp_dat['hr'].astype(float)

        #ap data. (NEEDS WORK)
        #ap can sometimes be 3 chars long, so above breaks
        '''
        kp_dat['ap1']= kp_dat['ap1'].astype(float)
        kp_dat['ap2']= kp_dat['ap2'].astype(float)
        kp_dat['ap2']= kp_dat['ap2'].mask(kp_dat['ap2'] == 1, np.nan)
        kp_dat['ap']=kp_dat['ap2'].fillna(kp_dat['ap1'])
        kp_dat = kp_dat.drop(columns=['ap1','ap2'])'''

        df_array.append(kp_dat)

    df_kp = pd.concat(df_array)
    df_kp = df_kp[['date','hr','Kp']].sort_values('date')

    
    return df_kp

kp_data = transform_kp(download_path)

kp_to_csv = str(download_path) +'/kp_data.csv'
kp_data.to_csv(kp_to_csv,index=False)
print(kp_data)
