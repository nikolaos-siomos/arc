#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 10:33:02 2022

@author: nick
"""

import sys, glob, os

sys.path.insert(0, '/home/nikos/Nextcloud/Programs/git/lidar_molecular/')

from lidar_molecular import raman_scattering, rayleigh_scattering, make_gas
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import us_std
from subprocess import run

temperature = np.arange(190.,315.,5.)


laser_wv = 354.717 #354.818 in vacuum
# laser_wv = 532.075 #532.223 in vacuum
# laser_wv = 1064.150 #1064.442 in vacuum

real_if = False

# fpath = '/home/nikos/Nextcloud/Rotational Raman Lines WG/Data_files/Filters/Alluxa_2623_ 530.2-2 OD6 Ultra Narrow BP, Lot 6-4042.dat'
# fpath = '/home/nikos/Nextcloud/Rotational Raman Lines WG/Data_files/Filters/Alluxa_3236_1056 BP OD6, Lot 12-0751.dat'
# fpath = '/home/nikos/Nextcloud/Rotational Raman Lines WG/Data_files/Filters/NBF_Filters_Datasheets/NBF_UV_RR.dat'
if real_if:
    if np.abs(laser_wv - 355.) < 5:
        pattern = 'NBF_UV_RR'
        # pattern = 'NBF_UV_RR_Theory'
    if np.abs(laser_wv - 532.) < 5:
        pattern = 'NBF_VIS_RR'
    if np.abs(laser_wv - 1064.) < 5:
        pattern = 'NBF_IR_RR'
    fpath = f'/home/nikos/Nextcloud/Rotational Raman Lines WG/Data_files/Filters/NBF_Filters_Datasheets/{pattern}.dat'


    if_data = np.loadtxt(fpath,skiprows=1)
    if_wave = if_data[:,0]
    if_tran = if_data[:,1]

    if_cwl = np.sum(if_wave*if_tran)/np.sum(if_tran)
    if_fwhm = np.max(if_wave[if_tran >= np.max(if_tran) / 2.]) - np.min(if_wave[if_tran >= np.max(if_tran) / 2.])
else:
    if_cwl = 355
    if_fwhm = 1.
    pattern = 'SIM_UV_RR'

aoi = np.arange(0., 3. + 0.05, 0.05)
aoi_theta = np.pi * aoi / 180.

n_air = 1.0003
n_if = 2.

wv_shift = if_cwl * np.sqrt(1. - (n_air/n_if)**2 * np.sin(aoi_theta)**2) - if_cwl

filters = np.empty((aoi.size),dtype = object)

for j in range(aoi.size):
    if real_if == True:
        filters[j] = raman_scattering.CustomFilter(wavelengths = if_wave + wv_shift[j], transmittances = if_tran)    
    else:
        filters[j] = raman_scattering.SquareFilter(wavelength = if_cwl + wv_shift[j], width = if_fwhm)  

ds_flt =  xr.DataArray(dims = ['aoi', 'temperature'], coords = [aoi, temperature])

ds_flt_fill = ds_flt.values.copy()

for j in range(temperature.size):
                    
    for l in range(aoi.size):
        
        rrb = raman_scattering.RotationalRaman(wavelength=laser_wv,max_J=101,temperature=temperature[j], optical_filter = filters[l]) 

        ds_flt_lj, xsections_flt_lj, bsc_flt_lj = rrb.rayleigh_cross_section()
        ds_flt_fill[l,j] = ds_flt_lj
        
ds_flt.values = ds_flt_fill
ds_diff_flt = ((ds_flt.copy() - ds_flt[0,0].copy())/ds_flt[0,0].copy())
ds_diff2_flt = ((ds_flt.copy() - ds_flt[0,:].copy())/ds_flt[0,:].copy())

dir_plt = f'./plots/{pattern}'
os.makedirs(dir_plt, exist_ok = True)

ds_flt.plot(cmap = 'jet')
plt.xlabel('Temperature [K]')
plt.ylabel('Angle of Incidence')
plt.title('$C_{bsc,m}$ [$m^{2} sr^{-1}$], $λ_ο$ = '+f'{laser_wv}nm')
plt.xlim(temperature[0],temperature[-1])
plt.ylim(aoi[0],aoi[-1])
if real_if:
    fpath = os.path.join(dir_plt,f'bsc_real_LW_{np.round(laser_wv,2)}nm_IF_{np.round(if_cwl,2)}nm_{np.round(if_fwhm,1)}nm.png')
    plt.savefig(fpath,dpi = 300)
else:
    fpath = os.path.join(dir_plt,f'bsc_square_LW_{np.round(laser_wv,2)}nm_IF_{np.round(if_cwl,2)}nm_{np.round(if_fwhm,1)}nm.png')
    plt.savefig(fpath,dpi = 300)
plt.show()
plt.close()


ds_diff_flt.plot(cmap='Spectral_r')#,levels = np.arange(-0.10,0.11,0.005))
plt.xlabel('Temperature [K]')
plt.ylabel('Angle of Incidence [${}^o C$]')
plt.xlim(temperature[0],temperature[-1])
plt.ylim(aoi[0],aoi[-1])
plt.title('$ΔC_{bsc,m}/C_{bsc,m,T_o,θ_o}$'+', '+'$T_o$ = '+f'{temperature[0]}K' + ', $θ_o$ = ' +f'{aoi[0]}'+'${}^o$')
if real_if:
    fpath = os.path.join(dir_plt,f'diff_real_LW_{np.round(laser_wv,2)}nm_IF_{np.round(if_cwl,2)}nm_{np.round(if_fwhm,1)}nm.png')
else:
    fpath = os.path.join(dir_plt,f'diff_square_LW_{np.round(laser_wv,2)}nm_IF_{np.round(if_cwl,2)}nm_{np.round(if_fwhm,1)}nm.png')
plt.savefig(fpath,dpi = 300)
plt.show()
plt.close()

ds_diff2_flt.plot(cmap='Spectral_r')#,levels = np.arange(-0.20,0.,0.005))
plt.xlabel('Temperature [K]')
plt.ylabel('Angle of Incidence [${}^o C$]')
plt.xlim(temperature[0],temperature[-1])
plt.ylim(aoi[0],aoi[-1])
plt.title('$ΔC_{bsc,m}/C_{bsc,m,T_i,θ_o}$'+', '+'$θ_o$ = ' +f'{aoi[0]}'+'${}^o$')
if real_if:
    fpath = os.path.join(dir_plt,f'diff2_real_LW_{np.round(laser_wv,2)}nm_IF_{np.round(if_cwl,2)}nm_{np.round(if_fwhm,1)}nm.png')
else:
    fpath = os.path.join(dir_plt,f'diff2_square_LW_{np.round(laser_wv,2)}nm_IF_{np.round(if_cwl,2)}nm_{np.round(if_fwhm,1)}nm.png')
plt.savefig(fpath,dpi = 300)
plt.show()
plt.close()

run(["convert", fpath, "-posterize", "8", fpath])

import pandas as pd

dir_lut = f'./lut/{pattern}'
os.makedirs(dir_lut, exist_ok = True)

if real_if:
    fpath = os.path.join(dir_lut,f'bsc_real_LW_{np.round(laser_wv,2)}nm_IF_{np.round(if_cwl,2)}nm_{np.round(if_fwhm,1)}nm.csv')
else:
    fpath = os.path.join(dir_lut,f'bsc_square_LW_{np.round(laser_wv,2)}nm_IF_{np.round(if_cwl,2)}nm_{np.round(if_fwhm,1)}nm.csv')
    
file = pd.DataFrame(data = ds_flt.values, columns = np.round(temperature,0), index = np.round(aoi,1))
file.to_csv(fpath)

if real_if:
    fpath = os.path.join(dir_lut,f'diff_real_LW_{np.round(laser_wv,2)}nm_IF_{np.round(if_cwl,2)}nm_{np.round(if_fwhm,1)}nm.csv')
else:
    fpath = os.path.join(dir_lut,f'diff_square_LW_{np.round(laser_wv,2)}nm_IF_{np.round(if_cwl,2)}nm_{np.round(if_fwhm,1)}nm.csv')

file = pd.DataFrame(data = ds_diff_flt.values, columns = np.round(temperature,0), index = np.round(aoi,1))
file.to_csv(fpath)

if real_if:
    fpath = os.path.join(dir_lut,f'diff2_real_LW_{np.round(laser_wv,2)}nm_IF_{np.round(if_cwl,2)}nm_{np.round(if_fwhm,1)}nm.csv')
else:
    fpath = os.path.join(dir_lut,f'diff2_square_LW_{np.round(laser_wv,2)}nm_IF_{np.round(if_cwl,2)}nm_{np.round(if_fwhm,1)}nm.csv')

file = pd.DataFrame(data = ds_diff2_flt.values, columns = np.round(temperature,0), index = np.round(aoi,1))
file.to_csv(fpath)

# for i in range(laser_wv.size):
#     plt.figure(figsize=(4,3))
#     # colors = ['tab:red','tab:blue','tab:orange']
#     for l in range(len(filter_t)):
#         plt.plot(ds_flt[l,2,i,:].values/ds_cab[2,i,:].values,altitude/1e3)
#     plt.legend(filter_t)
#     # for l in range(len(filter_t)):
#         # plt.fill_betweenx(altitude/1e3,x1=bsc_flt[l,1,i,:],x2=bsc_flt[l,3,i,:], alpha = 0.2)
#     plt.title('Temperature dependece per IFF')
#     plt.xlabel('$σ_{b,RR} / σ_{b,CAB}$')
#     plt.ylabel('Altitude [km]')
#     # plt.xlim([3.15*1E-31,3.25*1E-31])
#     # plt.xticks(np.arange(3.15*1E-31,3.25*1E-31, 2E-33))
#     plt.xscale('log')
#     plt.xlim([1E-4, 1.])
#     plt.ylim([0,20])
#     plt.grid()
#     plt.tight_layout()
#     plt.savefig(f'./norm_xs_{int(laser_wv[i])}.png',dpi=100)
#     plt.show()
    
# for i in range(laser_wv.size):
#     plt.figure(figsize=(4,3))
#     colors = ['tab:orange','tab:green']
#     lgd = []
#     for l in range(1,len(filter_t)):
#         plt.plot((ds_flt[l,2,i,:]/ds_flt[0,2,i,:])/(ds_flt[l,2,i,-1]/ds_flt[0,2,i,-1]).values,altitude/1e3,c=colors[l-1])
#         lgd.extend([f'{filter_t[l]}/{filter_t[0]}'])
#     plt.legend(lgd)
#     plt.title('Cross section ratio IFF_i/IFF_1')
#     plt.xlabel('$σ_{b,RR,i} / σ_{b,RR,1}$')
#     plt.ylabel('Altitude [km]')
#     plt.xlim([0.8, 3])
#     plt.ylim([0,20])
#     plt.grid()
#     plt.tight_layout()
#     plt.savefig(f'./temp_rdif_{int(laser_wv[i])}.png',dpi=100)
#     plt.show()