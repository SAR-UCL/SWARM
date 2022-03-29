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
#df = df[df['date'] == '2014-02-01']
#df = df[df['lat'].between(-35,35)]

#print(df)

def transform_df(df):

    def groupby(df, feat, func_n, rename_col):
        func = getattr(df.groupby(['date','p_num'], as_index=False)[feat], func_n)
        df = func().rename(columns={feat:rename_col}).drop(columns=['date','p_num'])
        
        return df

    #min
    ne_min = df.groupby(['date','p_num'], as_index=False)['Ne'].min().rename(
            columns={'Ne':'ne_min'}) #unique. Because need the date and
    ti_min = groupby(df,'Ti','min','ti_min')

    #max
    ne_max = groupby(df,'Ne', 'max', 'ne_max')
    ti_max = groupby(df, 'Ti','max', 'ti_max')

    #mean
    ne_mean = groupby(df, 'Ne', 'mean', 'ne_mean')
    ti_mean = groupby(df, 'Ti', 'mean', 'ti_mean')

    #stddev
    ne_std = groupby(df, 'Ne', 'std', 'ne_std')
    ti_std = groupby(df, 'Ti', 'std', 'ti_std')

    #EPB
    ibi_epb = groupby(df, 'b_ind', 'max', 'IBI')
    mssl_epb = groupby(df, 'sg_smooth','max','MSSL')

    df = pd.concat([ne_min, ne_max, ne_mean, ne_std, 
        ti_min, ti_max, ti_mean, ti_std,
        ibi_epb, mssl_epb],axis=1)

    df = df.sort_values(by=['date','p_num'], ascending=[True,True])

    ibi_count = df['IBI'].sum()
    mssl_count = df['MSSL'].sum()
    print(ibi_count)
    print(mssl_count)

    return df

df_gb = transform_df(df)
#df = df_gb.iloc[:15:]
print(df_gb)


figs, axs = plt.subplots(ncols=1, nrows=6, figsize=(7,5), 
dpi=90, sharex=True) #3.5 for single, #5.5 for double
axs = axs.flatten()

#df = df[df['lat'].between(-10,30)]

x = 'p_num'
hue = None

ax0y = 'ne_min'
sns.lineplot(ax = axs[0], data = df, x = x, y =ax0y, 
        palette = 'bone',hue = hue, legend=False)

ax1y = 'ne_max'
sns.lineplot(ax = axs[0], data = df, x =x, y =ax1y,
        palette = 'Set1', hue = hue, legend=False)

axs[0].set_yscale('log')

ax2y = 'ne_std'
sns.lineplot(ax = axs[1], data = df, x = x, y =ax2y, 
        palette = 'bone', hue = hue, legend = False)

ax3y = 'ti_min'
sns.lineplot(ax = axs[2], data = df, x = x, y =ax3y, 
        palette = 'Set2', hue = hue, legend = False)

ax3y = 'ti_max'
sns.lineplot(ax = axs[2], data = df, x = x, y =ax3y, 
        palette = 'Set2', hue = hue, legend = False)

ax3y = 'ti_std'
sns.lineplot(ax = axs[3], data = df, x = x, y =ax3y, 
        palette = 'Set2', hue = hue, legend = False)

ax4y = 'MSSL'
sns.lineplot(ax = axs[4], data = df, x = x, y =ax4y, 
        palette = 'rocket', hue = hue, legend = False)

ax5y = 'IBI'
sns.lineplot(ax = axs[5], data = df, x = x, y =ax5y, 
        palette = 'rocket', hue = hue, legend = False)
#axs[5].set_xscale('log')

plt.show()

def transpose_df(df):
    #df = df.groupby['date','p_num']
    None

    #df = df


