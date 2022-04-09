from ftplib import FTP
import pandas as pd
import os
import glob
from pathlib import Path
import numpy as np

dire = Path(r'/Users/sr2/OneDrive - University College ' 
    'London/PhD/Research/Models/dst/')


def transform_dst(dire):

    open_csv = str(dire) + '/dst_raw_14-21.csv'

    df = pd.read_csv(open_csv, on_bad_lines='skip', sep=" ", low_memory=False)

    #Change column names and merge dst
    df = df.rename(columns={'Unnamed: 0':'date','Format':'utc',
    'Unnamed: 9':'dst_1','Unnamed: 10':'dst_2','Unnamed: 11':'dst_3'})
    df['dst_4'] = df['dst_2'].fillna(df['dst_1'])
    df['dst'] = df['dst_3'].fillna(df['dst_4'])
    df = df[['date','utc','dst']]

    #remove headers
    date = df.pop('date')
    utc = df.pop('utc')
    df= df.apply(lambda x: pd.to_numeric(x, errors = 'coerce')).dropna()
    df['date'] = date
    df['utc'] = utc

    df['hr_d'] = df['utc'].str.slice(stop=2).astype(float)
    df=df.drop(columns=['utc'])

    return df

    #print(df)

dst_data = transform_dst(dire)
print(dst_data)

dst_to_csv = str(dire) +'/dst_proc_14-21.csv'
dst_data.to_csv(dst_to_csv,index=False)






