import geopandas
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import numpy as np

def geolocation():
    #path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/Phoenix/Instrument Data/analysis/Mar-21/data/Phoenix_TLE.csv'
    path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWENIX/data/densities-temps_130821.csv' 

    load_swenix = pd.read_csv(path)
    load_swenix = load_swenix.drop(columns=['swa_alt','notes'])
    #print(load_swenix)

    dtypes = load_swenix.dtypes
    #print('data types\n:', dtypes)

    #melt_swenix = pd.melt(load_swenix, id_vars=['date','swa_utc','pho_utc',], value_vars=['swa_lat','swa_long','pho_lat','pho_long'])
    #print(melt_swenix)

    #swarm = geopandas.GeoDataFrame(load_swenix, geometry = geopandas.points_from_xy(load_swenix.swa_long, load_swenix.swa_lat))
    swarm = geopandas.GeoDataFrame(load_swenix, geometry = geopandas.points_from_xy(load_swenix.swa_long, load_swenix.swa_lat))
    phoenix = geopandas.GeoDataFrame(load_swenix, geometry = geopandas.points_from_xy(load_swenix.pho_long, load_swenix.pho_lat))

    column = 'terminator'
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    ax = world.plot(color='white', edgecolor='black', figsize=(7.5, 3.5))
    #gdf.plot(ax=ax, column = column,# markersize = 20, legend = True, legend_kwds={'shrink': 0.75}, norm = LogNorm())
    #swarm.plot(ax=ax, markersize = 10, legend = True)
    phoenix.plot(ax=ax, markersize = 10, legend = True)

    #plt.title(f'Phoenix Geolocation \n Peak Counts by {column}')
    plt.tight_layout()
    plt.show()

    '''
    load_science = pd.read_csv(path)
    load_science = load_science.sort_values(['date', 'utc'], ascending =[True, True])
    #load_science = load_science.loc[load_science['date'] == '17/06/2018']
    print(load_science)

    
    gdf = geopandas.GeoDataFrame(load_science, geometry = geopandas.points_from_xy(load_science.long, load_science.lat))
    #gdf = gdf.loc[gdf.groupby('date')['sum'].idxmax()]

    column = 'STM'
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    ax = world.plot(color='white', edgecolor='black', figsize=(10, 5))
    #gdf.plot(ax=ax, column = column,# markersize = 20, legend = True, legend_kwds={'shrink': 0.75}, norm = LogNorm())
    gdf.plot(ax=ax, column = column, markersize = 20, legend = True)
    

    plt.title(f'Phoenix Geolocation \n Peak Counts by {column}')
    plt.tight_layout()
    plt.show()'''

geolocation()