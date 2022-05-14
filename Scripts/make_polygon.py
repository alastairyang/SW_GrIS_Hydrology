#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 16:26:33 2022

This script is a function that reads in the coordinates of a polygon and export
a Geojson file for QGIS processing

@author: donglaiyang
"""

import geopandas as gpd
from shapely.geometry import Polygon
import pandas as pd
from pyproj import Transformer

FilePath = "/Users/donglaiyang/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents_Alastair’s_MacBook_Pro/Buffalo/Courses/Remote Sensing/Project/ROI_latlon.csv"

data = pd.read_csv(FilePath)
lat_list = list(data['lat'])
lon_list = list(data['lon'])
polygon_geom = Polygon(zip(lon_list, lat_list))
crs = {'init':'epsg:4326'}
polygon_roi = gpd.GeoDataFrame(index=[0],crs=crs,geometry=[polygon_geom])
# write to file using geopandas's to_file function
polygon_roi.to_file(filename='ROI_polygon.geojson', driver='GeoJSON')
