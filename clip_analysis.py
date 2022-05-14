#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 18:20:45 2022

@author: donglaiyang
"""
import math
from glob import glob
# importing element tree
import xarray
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Transformer

def energy_balance(T,t):
    """This function calculates the net energy flux at an ice surface, given 
    the surface temperature
    
    T: temperautre
    t: time axia
    """
    lat       = 67.8 # degree latitude
    albedo    = 0.5 # non-dim
    cloudpass = 0.8 # non-dim, cloud lets 80% energy pass through
    T_d            = 5 # degree C, diurnal T range
    dTdz           = -0.0075 # degree C, air temperature lapse rate
    transmissivity = 0.65 # non-dim, atmospheric transmissivity
    z       = 1800 # m, mean elevation of the glacier
    z_cloud = 2000 # m, cloud base elevation
    S      = 1368 # W/m^2, solar constant
    sbc    = 5.67e-8 # stefan-boltzmann constant
    e_air  = 0.8 # non-dim, emissivity of GHG
    e_snow = 0.98 # non-dim, emissivity of glacier
    
    # get time
    hourinday, hours = day2hour(t)
    # replicate each element in T 24 times (for 24 hours every day)
    T = np.repeat(T,24)
    # calculate daily, seasonal, yearly variation
    T_daily = T + T_d/2*np.cos(math.pi*(12-hourinday)/12)
    
    sine_of_solar_angle = angle_solar_insolation(t, lat)
    Qsw_in = S*sine_of_solar_angle*cloudpass*transmissivity # shortwave in
    Qsw_out = albedo*Qsw_in # reflected back
    Qsw_net = Qsw_in - Qsw_out # sum up
    
    TCloudBase = T_daily + dTdz*(z_cloud-z) # cloud base temperature
    Qlw_in  = e_air*sbc*TCloudBase**4 # short wave in
    Tice = T_daily # replicate
    Tice[T_daily>273.15] = 273.15 # substitute above freezing point T by 0
    Qlw_out = e_snow*sbc*Tice**4  # longwave out radiation
    Qlw_net = Qlw_in - Qlw_out # Net Longwave Radiation
    #print(Qlw_net)
    
    # net energy flux
    Qnet = Qsw_net + Qlw_net
    return Qnet
    
def melt(Qnet):
    """This function estimates the amount of melt from net energy flux"""
    Lm     = 3.34*10**5 # J/kg, latent heat of melting
    rho    = 917 # kg/m^3, ice density
    return Qnet/(Lm*rho)

def angle_solar_insolation(t,lat):
    """This function calculates the  angle of solar insolation at this latitude
    
    t: time, in days
    """
    # get hours
    hourinday, hours = day2hour(t)
   
    solar_declination = 0.37
    
    sine_of_solar_angle = np.sin(lat*math.pi/180)*np.sin(solar_declination) + np.cos(lat*math.pi/180)*np.cos(solar_declination)*np.cos(math.pi*(12-hourinday)/12)
    sine_of_solar_angle = np.maximum(sine_of_solar_angle,np.zeros_like(sine_of_solar_angle))
    return sine_of_solar_angle
   
def day2hour(t):
    """This function convert string array of days to day array and repeating
    daily hour array"""
    dtime = 1/24; # one hour per day  
    hours = np.arange(1,len(t)+1,dtime)
    hourinday = hours%1*24
    return hourinday, hours   

#### MAIN SCRIPT ####
fieldname = "Ice_Surface_Temperature_Mean"
filesuffix = "*daily.v01.1.nc"
# resolve all sub-directories
AllPaths = glob("/Users/donglaiyang/Desktop/MODIS data 2019 SW GrIS/*/", recursive = True)  
N_folder = len(AllPaths)

# import each file
count = 0
july_t = []
for idx, path in enumerate(AllPaths):
    xmlpath = glob(path + "*.nc.xml")
    ncpath  = glob(path + filesuffix)
    if len(ncpath) == 0: # no daily file
        continue
    else: # get daily file
        # read xml and get the tree structure
        count = count + 1
        tree = ET.parse(xmlpath[0])
        root = tree.getroot() 
        # find the date specified in .xml
        ThisDate_elem = root.findall('.//RangeEndingDate')
        ThisDate = ThisDate_elem[0].text
        data = xarray.open_dataset(ncpath[0])
        # extract temperature and concat to a new dataset
        if count==1:
            T_july = data[fieldname]
            july_t.append(data[fieldname].attrs['data_date'])
        else:
            T_july = xarray.concat([T_july, data[fieldname]],dim='time')
            july_t.append(data[fieldname].attrs['data_date'])

# reassign time coordinate
T_july.coords['time'] = july_t
T_july = T_july.sortby('time')

# build coordinates
n_x = T_july.shape[2]
n_y = T_july.shape[1]
x_l = T_july.attrs['Proj_ul_xy'][0]
x_r = T_july.attrs['Proj_lr_xy'][0]
y_u = T_july.attrs['Proj_lr_xy'][1]
y_d = T_july.attrs['Proj_ul_xy'][1]
x = np.linspace(x_l,x_r,n_x)
y = np.linspace(y_d,y_u,n_y)
X,Y = np.meshgrid(x,y)

# reassign x and y coordinates
T_july.coords['y'] = y
T_july.coords['x'] = x
lon = np.array([-48.522126883035405,
                -47.819001883035405,
                -47.819001883035405, 
                -48.522126883035405])
lat = np.array([67.61074520312502,
                67.61074520312502,
                67.87601435409634,
                67.87704872715982])
# transform the coordinates
transformer = Transformer.from_crs("epsg:4326", "epsg:3411")
x1, y1 = transformer.transform(lat, lon)

# create a rectangular mask
mask_all = np.zeros((n_y,n_x))
y_mask = np.where(np.logical_and(Y>y1.min(), Y<y1.max()), 1, 0)
x_mask = np.where(np.logical_and(X>x1.min(), X<x1.max()), 1, 0)
mask = np.where(y_mask+x_mask==2, True, False)

# extract data
T_extract = np.zeros((len(july_t),mask.sum()))
for idx, day in enumerate(july_t):
    T_extract[idx,:] = T_july.sel(time=day).values[mask]

# replace the out-of-range (usually clouds) by the column means
abnormal_T = np.logical_or(T_extract<243, T_extract>280)
T_extract[abnormal_T] = math.nan;
col_mean = np.nanmean(T_extract, axis=0)
#Find indices that you need to replace
inds = np.where(np.isnan(T_extract))
#Place column means in the indices. Align the arrays using take
T_extract[inds] = np.take(col_mean, inds[1])

# make line plot
fig, ax = plt.subplots()
ax.plot(T_july.coords['time'].values, T_extract)    
ax.set_ylabel('Temperature at each pixel')
ax.set_xticklabels(T_july.coords['time'].values, rotation=90)
plt.savefig('/Users/donglaiyang/Desktop/Temperatures.png', format='png',dpi=350)

#%%
##### Surface mass balance #####
# area
dxdy = abs(x[1]-x[0])*abs(y[1]-y[0])
# get 
melt_t = np.zeros((T_extract.shape[0]*24,T_extract.shape[1]))
for sample in range(T_extract.shape[1]):
    Qnet = energy_balance(T_extract[:,sample], july_t)
    melt_t[:,sample] = melt(Qnet)

# calculate the total melt for each pixel
melt_pixel = np.cumsum(melt_t, axis=0)*dxdy

# calculate the total melt in this region
melt_sum = np.sum(melt_pixel, axis=1)

# calculate the normalized melt
n_pixel = T_extract.shape[1]
area = dxdy*n_pixel
melt_norm = melt_sum/area

#%% Make plots of melt time series
hourinday, hours = day2hour(july_t)
fig, ax = plt.subplots(3,1)
fig.set_size_inches(10.5, 10.5)
ax[0].plot(hours, melt_t, linewidth=0.2, alpha=0.3)
ax[0].set_ylabel('melt volume per area, m^3')
ax[1].plot(hours, melt_norm)
ax[1].set_ylabel('cumulative melt per area, m^3')
ax[2].plot(hours, melt_sum)
ax[2].set_ylabel('cumul. melt for the whole region, m^3')
ax[2].set_xlabel('days')
plt.savefig('/Users/donglaiyang/Desktop/melt.png', format='png',dpi=350)


