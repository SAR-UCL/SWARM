import pandas as pd
from pathlib import Path
from datetime import date
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import seaborn as sns
pd.set_option('display.max_rows',  10) #or 10 or None

path = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/'
        'Missions/SWARM/Non-Flight Data/Analysis/Mar-22/data/solar_max/')

dir_suffix = '2014'
pre_classified = str(path) + '/' + dir_suffix +'-data-2022-03-03.csv'
post_classified = str(path) + '/ml_model/SG-filtered_14-15.csv'

def open_all(filename):
    print('Loading data...')
    df = pd.read_csv(filename)
    return df
#df = open_all(load_all)
df = open_all(post_classified)

#print(df)

def transform_df(df):

    def min_gb(df, feat, rename_col):
        df = df.groupby(['date','p_num'], as_index=False)[feat].min().rename(
                columns={feat:rename_col}).drop(columns=['date','p_num'])
        return df

    def max_gb(df, feat, rename_col):
        df = df.groupby(['date','p_num'], as_index=False)[feat].max().rename(
                columns={feat:rename_col}).drop(columns=['date','p_num'])
        return df

    def mean_gb(df, feat, rename_col):
        df = df.groupby(['date','p_num'], as_index=False)[feat].mean().rename(
                columns={feat:rename_col}).drop(columns=['date','p_num'])
        return df

    def std_gb(df, feat, rename_col):
        df = df.groupby(['date','p_num'], as_index=False)[feat].std().rename(
                columns={feat:rename_col}).drop(columns=['date','p_num'])
        return df
    

    #min
    ne_min = df.groupby(['date','p_num'], as_index=False)['Ne'].min().rename(columns={'Ne':'Ne_min'})
    ti_min = min_gb(df,'Ti','ti_min')

    #max
    ne_max = max_gb(df,'Ne', 'ne_max')
    ti_max = max_gb(df, 'Ti', 'ti_max')

    #mean
    ne_mean = mean_gb(df, 'Ne', 'ne_mean')
    ti_mean = mean_gb(df, 'Ti', 'ti_mean')

    #stddev
    ne_std = std_gb(df, 'Ne', 'ne_std')
    ti_std = std_gb(df, 'Ti', 'ti_std')

    #pot_min = min_gb(df,'Ne','Ne_min')

    '''
    d_max = df.groupby(['date','p_num'], as_index=False)['Ne'].max().rename(columns={'Ne':'Ne_max'})
    d_min = df.groupby(['date','p_num'], as_index=False)['Ne'].min().rename(columns={'Ne':'Ne_min'}).drop(columns=['date','p_num'])
    d_avg = df.groupby(['date','p_num'], as_index=False)['Ne'].mean().rename(columns={'Ne':'Ne_avg'}).drop(columns=['date','p_num'])
    d_std = df.groupby(['date','p_num'], as_index=False)['Ne'].max().rename(columns={'Ne':'Ne_std'}).drop(columns=['date','p_num'])
    ibi_epb = df.groupby(['date','p_num'], as_index=False)['b_ind'].max().rename(columns={'b_ind':'IBI'}).drop(columns=['date','p_num'])
    mssl_epb = df.groupby(['date','p_num'], as_index=False)['sg_smooth'].max().rename(columns={'sg_smooth':'MSSL'}).drop(columns=['date','p_num'])
    '''

    #df = pd.concat([d_max, d_min, d_avg, d_std, ibi_epb, mssl_epb], axis=1)
    #df['Ne_std'] = d_sd
    df = pd.concat([ne_min, ne_max, ne_mean, ne_std, 
             ti_min, ti_max, ti_mean, ti_std],axis=1)

    df = df.sort_values(by=['date','p_num'], ascending=[True,True])

    #ibi_count = df['IBI'].sum()
    #mssl_count = df['MSSL'].sum()
    #print(ibi_count)
    #print(mssl_count)

    return df

df_gb = transform_df(df)
print(df_gb)