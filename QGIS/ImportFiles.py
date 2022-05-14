#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 18:20:45 2022

@author: donglaiyang
"""

import os
from glob import glob
# importing element tree
import xml.etree.ElementTree as ET

def get_HHMM(filestr, tail):
    """ Get the hour and minute (HHMM) of the nc file"""
    len_tail = len(tail)
    len_HHMM = 4
    return filestr[(-1*len_tail - len_HHMM):-1*len_tail]

def push_qgis(FilePath, LayerName):
    """ create raster layer, set crs, and push to QGIS"""
    rlayer = QgsRasterLayer(FilePath, LayerName)
    rlayer.setCrs( QgsCoordinateReferenceSystem(3411, QgsCoordinateReferenceSystem.EpsgCrsId) )
    # push to QGIS
    QgsProject.instance().addMapLayers([rlayer])
    

#### MAIN SCRIPT ####
# resolve all sub-directories
AllPaths = glob("/Users/donglaiyang/Desktop/MODIS data 2019 SW GrIS/*/", recursive = True)  
N_folder = len(AllPaths)

# import each file via iteration
for idx, path in enumerate(AllPaths):
    xmlpath = glob(path + "*.nc.xml")
    ncpath  = glob(path + "*daily.v01.1.nc")
    if len(ncpath) == 0: # no daily file
        continue
    else: # get daily file
        # read xml and get the tree structure
        tree = ET.parse(xmlpath[0])
        root = tree.getroot() 
        # find the date specified in .xml
        ThisDate_elem = root.findall('.//RangeEndingDate')
        ThisDate = ThisDate_elem[0].text
        # FileName
        LayerName = ThisDate + "_" + get_HHMM(ncpath[0], ".swath.v01.1.nc")
        FilePath = 'NETCDF:"' + ncpath[0] + '":Ice_Surface_Temperature_Mean'
        
        push_qgis(FilePath, LayerName)

