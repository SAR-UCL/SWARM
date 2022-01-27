

#https://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html

#https://chart-studio.plotly.com/~empet/14813.embed


import netCDF4 as nc
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with nc.netcdf_file('compday.KsNOdsJTz1.nc', 'r') as f:
        lon = f.variables['lon'][::]       # copy longitude as list
        lat = f.variables['lat'][::-1]     # invert the latitude vector -> South to North
        olr = f.variables['olr'][0,::-1,:] # olr= outgoing longwave radiation
    f.fp   