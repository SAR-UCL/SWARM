
import scipy
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm

#Loading and exporting
path = r'/Volumes/swarm-diss.eo.esa.int/Level2daily/Latest_baselines/IBI/TMS/Sat_A'
#path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Instrument Data/Analysis/Sept-21/data/'
file_name = r'IPD-Dayside-Cleaned.csv'
load_csv = path + file_name
load_csv = pd.read_csv(load_csv)
#print(load_csv)

#Filter data for EPB observation
#\cite{Park2013} See paper for info
#explore_data = load_csv.loc[load_csv['hemi'] == 'night']
#explore_data = explore_data[explore_data['lat'].between(-45,45)]
#explore_data = load_csv.loc[load_csv['bubble'] == 1] #1 Confirmed Bubble, 0 unconfirmed bubble, -1 unanalyzable bubble

print(explore_data)
