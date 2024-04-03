#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 11:11:29 2023

@author: nikos
"""

import sys
sys.path.insert(0, '/home/nikos/Nextcloud/Programs/git/lidar_molecular/')

from lidar_molecular import raman_scattering
import xarray as xr
import us_std
import numpy as np
from scipy import interpolate


def get_ds(height, laser_wv, istotal = False, ds_type = 'astokes'):

    # temperature: The atmospheric temperature according to US Standard Atmosphere
    atm = us_std.Atmosphere()
    temperature = np.array([atm.temperature(z) for z in 1e3 * height])

    for k in range(temperature.size):
        rrb = raman_scattering.RotationalRaman(wavelength=laser_wv,max_J=41,temperature=temperature[k], istotal = False)
        if k == 0:
            if ds_type == 'astokes':
                dl_N2 = rrb.dl_astokes_N2
                dl_O2 = rrb.dl_astokes_O2
                dl_CO2 = rrb.dl_astokes_CO2
            elif ds_type == 'stokes':
                dl_N2 = rrb.dl_stokes_N2
                dl_O2 = rrb.dl_stokes_O2
                dl_CO2 = rrb.dl_stokes_CO2       
            else:
                raise Exception(f'Unrecognized ds_type: {ds_type}. Please select one of: stokes, astokes')
                
            ds_N2 = xr.DataArray(dims = ('height', 'wavelength'),
                                 coords = (height, dl_N2))
            ds_O2 = xr.DataArray(dims = ('height', 'wavelength'),
                                 coords = (height, dl_O2))
            ds_CO2 = xr.DataArray(dims = ('height', 'wavelength'),
                                 coords = (height, dl_CO2))

        if ds_type == 'astokes':
            ds_N2[k,:] = rrb.ds_astokes_N2
            ds_O2[k,:] = rrb.ds_astokes_O2
            ds_CO2[k,:] = rrb.ds_astokes_CO2    
        elif ds_type == 'stokes':
            ds_N2[k,:] = rrb.ds_stokes_N2
            ds_O2[k,:] = rrb.ds_stokes_O2
            ds_CO2[k,:] = rrb.ds_stokes_CO2    
        else:
            raise Exception(f'Unrecognized ds_type: {ds_type}. Please select one of: stokes, astokes')
            
        ds_N2 = ds_N2.where(ds_N2 > 0)
        ds_O2 = ds_O2.where(ds_O2 > 0)
        ds_CO2 = ds_CO2.where(ds_CO2 > 0)

    return(ds_N2, ds_O2, ds_CO2)

def get_if_transmission(fpath, wave_lims):
    
    # Read the filter data. if_wave: the wavelength array, if_tran: the transmission array
    if_data = np.loadtxt(fpath,skiprows=1)
    if_wave_raw = if_data[:,0]
    if_tran_raw = if_data[:,1]
    
    # Sort the wavelength and transmission arrays with ascending wavelength
    sorting = if_wave_raw.argsort()
    if_wave_raw = if_wave_raw[sorting]
    if_tran_raw = if_tran_raw[sorting]
    
    # Add a 50nm buffer zone of 0 transmission for wavelengths smaller than the minimum wavelength of the filter
    buffer = np.arange(if_wave_raw[0]-20., if_wave_raw[0], 0.02)
    if_wave = np.hstack((buffer,if_wave_raw))
    if_tran = np.hstack((np.zeros(buffer.size),if_tran_raw))
    
    # Recreate the filter transmission with high resolution to avoid 
    temp_transmission = interpolate.interp1d(if_wave, if_tran, bounds_error = False, fill_value = 0.)
    if_wave_hq = np.arange(wave_lims[0], wave_lims[-1] + 0.01, 0.01)
    if_tran_hq = temp_transmission(if_wave_hq)

    # if_cwl: The equivalent central wavelength of the filter
    if_cwl = np.sum(if_wave*if_tran)/np.sum(if_tran)

    # # if_fwhm: The equivalent bandwidth of the filter
    # if_fwhm = np.max(if_wave[if_tran >= np.max(if_tran) / 2.]) - np.min(if_wave[if_tran >= np.max(if_tran) / 2.])

    # aoi: A range of angle of incidence at the IF in degrees. Used to constract a 2D function of the transmission (wavelength,aoi)
    aoi = np.arange(0., 3. + 0.01, 0.01)
    aoi_theta = np.pi * aoi / 180.

    # n_air: Refractive Index of air
    n_air = 1.0003

    # n_if: Refractive Index of the IF
    n_if = 2.

    transmission_1D = interpolate.interp1d(if_wave_hq, if_tran_hq, bounds_error = False, fill_value = 0.)
    print(if_cwl)
    # wv_shift: The wavelength shift of the transmission of the IF due to the angle of incidence
    wv_shift = if_cwl * np.sqrt(1. - (n_air/n_if)**2 * np.sin(aoi_theta)**2) - if_cwl

    if_tran_2D = np.nan * np.zeros((aoi.size, if_wave_hq.size))
    for j in range(aoi.size):
        if_tran_2D[j,:] = transmission_1D(if_wave_hq - wv_shift[j])
         
    transmission_2D = xr.DataArray(if_tran_2D, 
                                   dims = ('aoi','wavelength'),
                                   coords = [aoi,if_wave_hq])
    
    return(transmission_2D)

def aggregate_cross_sections_3D(aoi, transmission, ds_N2, ds_O2, ds_CO2):
     
    c_N2 = 0.780796
    c_O2 = 0.209448
    c_CO2 = 0.000416
    
    ds = xr.DataArray(dims = aoi.dims,
                      coords = aoi.coords)
    
    ds_zero_aoi = xr.DataArray(dims = aoi.mean('rays').dims,
                               coords = aoi.mean('rays').coords)
    
    for i in range(aoi.theta_x.size):
        for j in range(aoi.theta_y.size):
            for k in range(aoi.height.size):
                ds_N2_zero_aoi = (ds_N2[k,:] * transmission.loc[0,:].interp(wavelength = ds_N2.wavelength.values)).sum(dim = 'wavelength').values    
                ds_O2_zero_aoi = (ds_O2[k,:] * transmission.loc[0,:].interp(wavelength = ds_O2.wavelength.values)).sum(dim = 'wavelength').values    
                ds_CO2_zero_aoi = (ds_CO2[k,:] * transmission.loc[0,:].interp(wavelength = ds_CO2.wavelength.values)).sum(dim = 'wavelength').values    

                ds_zero_aoi[i,j,k] = (c_N2 * ds_N2_zero_aoi + c_O2 * ds_O2_zero_aoi + c_CO2 * ds_CO2_zero_aoi) / (c_N2 + c_O2 + c_CO2)
              
                ds_N2_tot = (ds_N2[k,:] * transmission.interp(aoi = np.abs(aoi[i,j,k,:])).interp(wavelength = ds_N2.wavelength.values)).sum(dim = 'wavelength')
                ds_O2_tot = (ds_O2[k,:] * transmission.interp(aoi = np.abs(aoi[i,j,k,:])).interp(wavelength = ds_O2.wavelength.values)).sum(dim = 'wavelength')
                ds_CO2_tot = (ds_CO2[k,:] * transmission.interp(aoi = np.abs(aoi[i,j,k,:])).interp(wavelength = ds_CO2.wavelength.values)).sum(dim = 'wavelength')

                ds[i,j,k,:] = (c_N2 * ds_N2_tot + c_O2 * ds_O2_tot + c_CO2 * ds_CO2_tot) / (c_N2 + c_O2 + c_CO2)
            
    ds_mean = ds[:,:,:,1:].mean(dim = 'rays')

    ds_single = ds[:,:,:,0]
    
    return(ds, ds_mean, ds_single, ds_zero_aoi)

def aggregate_cross_sections_1D(aoi, transmission, ds_N2, ds_O2, ds_CO2):
     
    c_N2 = 0.780796
    c_O2 = 0.209448
    c_CO2 = 0.000416
    
    ds = xr.DataArray(dims = aoi.dims,
                      coords = aoi.coords)
    
    ds_zero_aoi = xr.DataArray(dims = aoi.mean('rays').dims,
                               coords = aoi.mean('rays').coords)

    for k in range(aoi.height.size):
        ds_N2_zero_aoi = (ds_N2[k,:] * transmission.loc[0.,:].interp(wavelength = ds_N2.wavelength.values)).sum(dim = 'wavelength').values    
        ds_O2_zero_aoi = (ds_O2[k,:] * transmission.loc[0.,:].interp(wavelength = ds_O2.wavelength.values)).sum(dim = 'wavelength').values    
        ds_CO2_zero_aoi = (ds_CO2[k,:] * transmission.loc[0.,:].interp(wavelength = ds_CO2.wavelength.values)).sum(dim = 'wavelength').values    

        ds_zero_aoi[k] = (c_N2 * ds_N2_zero_aoi + c_O2 * ds_O2_zero_aoi + c_CO2 * ds_CO2_zero_aoi) / (c_N2 + c_O2 + c_CO2)
      
        ds_N2_tot = (ds_N2[k,:] * transmission.interp(aoi = np.abs(aoi[k,:])).interp(wavelength = ds_N2.wavelength.values)).sum(dim = 'wavelength')
        ds_O2_tot = (ds_O2[k,:] * transmission.interp(aoi = np.abs(aoi[k,:])).interp(wavelength = ds_O2.wavelength.values)).sum(dim = 'wavelength')
        ds_CO2_tot = (ds_CO2[k,:] * transmission.interp(aoi = np.abs(aoi[k,:])).interp(wavelength = ds_CO2.wavelength.values)).sum(dim = 'wavelength')

        ds[k,:] = (c_N2 * ds_N2_tot + c_O2 * ds_O2_tot + c_CO2 * ds_CO2_tot) / (c_N2 + c_O2 + c_CO2)
        
    ds_mean = ds.mean(dim = 'rays')
    
    return(ds, ds_mean, ds_zero_aoi)

def molecular_bsc_coef(fpath, height, laser_wv, istotal = False):
    
    atm = us_std.Atmosphere()
    # temperature: The atmospheric temperature according to US Standard Atmosphere
    temperature = np.array([atm.temperature(z) for z in 1e3 * height])
    # pressure: The atmospheric pressure according to US Standard Atmosphere
    pressure = np.array([atm.pressure(z) for z in 1e3 * height])

    bsc_mol_flt =  xr.DataArray(dims = ['height'], coords = [height])

    for k in range(temperature.size):
        rrb = raman_scattering.RotationalRaman(wavelength=laser_wv,max_J=101,
                                               temperature=temperature[k], 
                                               istotal = False)
        
        _, _, bsc_flt_k = rrb.cabannes_cross_section(pressure = pressure[k])
        bsc_mol_flt[k] = bsc_flt_k

    return(bsc_mol_flt)
