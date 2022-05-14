#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 10:55:57 2022

@author: donglaiyang
"""

import math
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import rasterio.mask
import rasterio.plot

def lake_volume(tot_area, pix_area, data, n_MC):
    """ Use Monte Carlo to estimate the lake volume """
    mean = np.nanmean(data)
    std  = np.nanstd(data)
    s = np.random.normal(mean, std, n_MC)
    # pixal axrea * lake depth
    sum_observed = np.nansum(data)*pix_area
    sum_inferred = (tot_area - pix_area*n_finitepix(data))*s
    tot_V = sum_observed + sum_inferred
    
    return tot_V

def n_finitepix(data):
    """count the number of finite values in the matrix"""
    return len(~np.isnan(data))

def add_errorprop(error):
    """error propagation for addition"""
    return np.sqrt(np.sum(error**2))

def linear_estimate(a,b,data):
    """Get the linear estimate of a given data and linear model"""
    return a*data+b

raster_path = '/Users/donglaiyang/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents_Alastair’s_MacBook_Pro/Buffalo/Courses/Remote Sensing/Project/lake_raster_naureen_clips.tif'
vector_path = '/Users/donglaiyang/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents_Alastair’s_MacBook_Pro/Buffalo/Courses/Remote Sensing/Project/lake_Geojson.geojson'
r_data = rasterio.open(raster_path) # this does NOT READ DATA!! You need .read later
bounds = r_data.bounds
# build the coordinates; bound order is left-bottom, right-top
extent=[bounds[0], bounds[2], bounds[1], bounds[3]]
dxdy = abs(bounds[2]-bounds[0])/r_data.shape[1]
# Reads in as a numpy array
r_data_array = np.squeeze(r_data.read(), axis=0)
r_data_array[r_data_array < 0.1] = math.nan
fig, ax = plt.subplots()
rasterio.plot.show(r_data_array, extent = extent, cmap='terrain')

## import the vector data
polyg_data = gpd.read_file(vector_path)

# Monte Carl
n_MC = 1000
lake_indices = [0,1,2,3,4,5,6,7,8,10]
volumes = np.zeros((len(lake_indices), n_MC))
for idx, lake_idx in enumerate(lake_indices):
    lake_bool = rasterio.mask.raster_geometry_mask(r_data, 
                                                   polyg_data.geometry[idx],
                                                   invert=True)
    lake = r_data_array[lake_bool[0]]
    volumes[idx,:] = lake_volume(polyg_data.loc[lake_idx].geometry.area, 
                                 dxdy, lake, n_MC)
# make a scatter plot
# lake volume vs area
# first, mean of all estimated volume estimations
vol_means = np.mean(volumes, axis=1)
vol_stds  = np.std(volumes, axis=1)

fig, ax = plt.subplots()
areas = polyg_data.loc[lake_indices].geometry.area
ax.errorbar(areas, vol_means,
            vol_stds, fmt='o')
ax.set_xlabel('Lake area (m^2)')
ax.set_ylabel('Lake volume (m^3)')
# add a line of best fit
a, b = np.polyfit(areas, vol_means, 1)
area_linear = np.arange(np.min(areas),np.max(areas),100)
plt.plot(area_linear, a*area_linear+b, color='red',linestyle='-.')

# error as a function of area
c, d = np.polyfit(areas, vol_stds, 1)
# get inferred lake volume and inferred error
# first find the rest of lake indices
otherlakes_indices = set(polyg_data.index) - set(lake_indices)
otherareas = polyg_data.loc[otherlakes_indices].geometry.area
estimate_volume = np.array(linear_estimate(a, b, otherareas))
estimate_errors = np.array(linear_estimate(c, d, otherareas))

# Total lake volume and error 
tot_volume = np.sum(vol_means) + np.sum(estimate_volume)
tot_errors = add_errorprop(np.concatenate((vol_stds, estimate_errors)))

    
    
