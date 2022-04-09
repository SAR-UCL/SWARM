import os
import datetime
from pathlib import Path
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

#Swarm data
swarm_dir = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/'
        'Missions/SWARM/Non-Flight Data/Analysis/Mar-22/data/solar_max/')
swarm_data = str(swarm_dir) + '/ml_model/SG-filtered_14-15.csv'

kp_dir = Path(r'/Users/sr2/OneDrive - University College ' 
    'London/PhD/Research/Models/kp/')
kp_data = str(kp_dir) +'/kp_data.csv'

dst_dir = Path(r'/Users/sr2/OneDrive - University College ' 
    'London/PhD/Research/Models/dst/')
dst_data = str(dst_dir) +'/dst_proc_14-21.csv'

def open_all(filename):
    print('Loading data...')
    df = pd.read_csv(filename)
    return df

def join_swarm_kp():

    #Get and sort swarm data
    df_swarm = open_all(swarm_data)

    get_hr = df_swarm['utc'].str.slice(stop=2).astype(float)

    def round_down(num):
        return num - (num%3)

    df_swarm['hr'] = get_hr.apply(round_down)

    #get kp data
    df_kp = open_all(kp_data)

    #join the dataframes
    df = df_swarm.merge(df_kp, on=['date','hr'])

    return df

df_swarm = join_swarm_kp()

def join_swarm_dst():

    #Get and sort swarm data
    #df_swarm = open_all(swarm_data)
    
    get_hr_d = df_swarm['utc'].str.slice(stop=2).astype(float)

    def round_down(num):
        return num - (num%1)
    df_swarm['hr_d'] = get_hr_d.apply(round_down)
    #print(df_swarm)

    #get kp data
    df_dst = open_all(dst_data)

    #join the dataframes
    df = df_swarm.merge(df_dst, on=['date','hr_d'])

    return df

df = join_swarm_dst()

def final_clean(df):

    df = df.drop(columns=['hr','hr_d'])

    return df

df= final_clean(df)

print(df)