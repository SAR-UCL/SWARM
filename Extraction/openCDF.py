import cdflib
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import numpy as np
import glob
from pathlib import Path
import geopandas

TII_dir = Path(r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Data/nov-21/TII/demo')


def openTII(dire):
    cdf_array = []
    cdf_files = dire.glob('*.cdf')
    for f in cdf_files:
        cdf = cdflib.CDF(f) #asign to cdf object   

        #Scalars Vars
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

        #convert to df 
        cdf_df = pd.DataFrame({"Timestamp":utc,"alh":al_h,"alv":al_v, "xth":xt_h,"xtv":xt_v,
            "alh_e":al_h_err,"alv_e":al_v_err, "xth_e":xt_h_err,"xtv_e":xt_v_err})
        cdf_array.append(cdf_df)

        tii_data = pd.concat(cdf_array)
    
    return tii_data

tii_data = openTII(TII_dir)

def convert2Datetime(utc):
    #https://pypi.org/project/cdflib/
    utc = cdflib.epochs.CDFepoch.to_datetime(utc)
    return utc

tii_data['datetime'] = tii_data['Timestamp'].apply(convert2Datetime).str[0].astype(str)

print(tii_data)