#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 17:42:47 2023

@author: nikos
"""
import xarray as xr
import numpy as np

def aerosol_bsc_coef(height, height_nodes, bsc_nodes):
    
    bsc_aer = xr.DataArray(bsc_nodes,
                           dims = ['height'], 
                           coords = [height_nodes])
    
    bsc_aer = bsc_aer.interp(height = height)

    return(bsc_aer)

def extinction_error(ds_real, ds_zero_aoi, hwin):
    
    ext_err = xr.DataArray(dims = ds_real.dims, coords = ds_real.coords)

    for k in range(ds_real.height.size):
        if k >= hwin and k < ds_real.height.size - 2:
            ext_err[k] = (1e3 * 0.5 * np.log(ds_zero_aoi[k-hwin:k+hwin+1]/ds_real[k-hwin:k+hwin+1])).polyfit('height',deg=1).polyfit_coefficients.values[0]
    # ext_err = (1e3 * 0.5 * np.log(ds_zero_aoi/ds_mean)).differentiate('height')

    return(ext_err)

def backscatter_error(ds_mean, ds_zero_aoi, bsc_aer, bsc_mol):
    
    bsc_err = 1e6 * (1. - ds_mean / ds_zero_aoi) * (bsc_aer + bsc_mol)

    return(bsc_err)

def cross_section_error(ds_mean, ds_zero_aoi):
    
    xcs_var = ((ds_mean - ds_zero_aoi)/ds_zero_aoi)

    return(xcs_var)