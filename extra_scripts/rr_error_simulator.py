#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 17:11:22 2023

@author: nikos
"""

import numpy as np
from matplotlib import pyplot as plt
from functions.angle_calculator import angle_of_incidence, angle_of_incidence_IF, change_tilt
from functions.rr_calculator import get_ds, get_if_transmission, aggregate_cross_sections_1D, molecular_bsc_coef
from functions.miscellaneous import aerosol_bsc_coef, extinction_error, backscatter_error, cross_section_error
import xarray as xr
import os, io

# -----------------------------------------------------------------------------
# Input variables
# -----------------------------------------------------------------------------
# laser_wv: The laser wavelength in nm in air
# laser_wv = 354.717 #354.818 in vacuum
# laser_wv = 532.075 #532.223 in vacuum
laser_wv = 1064.150 #1064.442 in vacuum


# fpath: Path to the IF ascii file
if np.abs(laser_wv - 355.) < 5:
    # use_filter = 'NBF_UV_RR'
    use_filter = 'NBF_UV_RR_Theory'
    # use_filter = 'NBF_UV_RR_Theory_edit'
    # use_filter = 'NBF_UV_RR_Theory_edit2'
    # use_filter = 'NBF_UV_RR_Theory_edit3'
    # use_filter = 'NBF_UV_RR_2filters'
    fpath = f'/home/nikos/Nextcloud/Rotational Raman Lines WG/Data_files/Filters/NBF_Filters_Datasheets/{use_filter}.dat'
    # pattern = 'DK300_300m_355-532_EO85878_355nm_coax_incl_parangles' #coaxial
    pattern = 'DK300_300m_355-532_EO85878_cut_355nm_incl_parangles' #biaxial - mid focus
    # pattern = 'DK300_300m_NR_355-532_EO85878_cut_355nm_incl_parangles' #biaxial - near focus
    # pattern = 'AC50_F150_VF_cut_opt355nm_incl_parangles' #biaxial - near rangle telescope - VF
    rpath = f'./ZEMAX_simulations/updated/{pattern}.dat'
if np.abs(laser_wv - 532.) < 5:
    use_filter = 'NBF_VIS_RR'
    fpath = f'/home/nikos/Nextcloud/Rotational Raman Lines WG/Data_files/Filters/NBF_Filters_Datasheets/{use_filter}.dat'
    # pattern = 'DK300_300m_355-532_EO85878_532nm_coax_incl_parangles' #coaxial
    pattern = 'DK300_300m_355-532_EO85878_cut_532nm_incl_parangles' #biaxial - mid focus
    # pattern = 'DK300_300m_NR_355-532_EO85878_cut_532nm_incl_parangles' #biaxial - near focus
    # pattern = 'AC50_F150_VF_cut_opt532nm_incl_parangles' #biaxial - near rangle telescope - VF
    rpath = f'./ZEMAX_simulations/updated/{pattern}.dat'
if np.abs(laser_wv - 1064.) < 5:
    use_filter = 'Alluxa_IR_RR_3236_P6'
    # fpath = f'/home/nikos/Nextcloud/Rotational Raman Lines WG/Data_files/Filters/NBF_Filters_Datasheets/{use_filter}.dat'
    fpath = f'/home/nikos/Nextcloud/Rotational Raman Lines WG/Data_files/Filters/Alluxa_Filters_Datasheets/{use_filter}.dat'
    pattern = 'DK300_300m_1064_EO47387_incl_parangles'
    rpath = f'./ZEMAX_simulations/updated/{pattern}.dat'

D = 150.
theta_z = 0.6 # in mrad
theta_t = 0.1 # laser tilt in mrad
M = 3.

file = io.open(rpath,'r',encoding='utf-16')
rdata = np.char.strip(np.loadtxt(file,delimiter=',',dtype=str))[:,:-1].astype(float)

height = 1e-3 * rdata[0,:] 
omega_if = 2. * xr.DataArray(180. * rdata[1:,:].T / np.pi, 
                        dims = ['height','rays'],
                        coords = [height, np.arange(rdata.shape[0]-1)])

# # Re-tilt the laser beam
# omega_if = change_tilt(omega = omega_if.copy(), theta_t = theta_t, 
#                         theta_z = theta_z, D = 150., M = M)

# ds_N2, ds_O2, ds_CO2: The Anti-stokes branch backscattering cross-sections of N2, O2, and CO2 respectively, in m2 sr-1
ds_N2, ds_O2, ds_CO2 = get_ds(height, laser_wv, istotal = False, ds_type = 'astokes')
ds_stokes_N2, ds_stokes_O2, ds_stokes_CO2 = \
    get_ds(height, laser_wv, istotal = False, ds_type = 'stokes')

# Find the RR line wavelenght limits
wave_max = np.max([np.max(ds_N2.wavelength), np.max(ds_O2.wavelength), np.max(ds_CO2.wavelength)])
wave_min = np.min([np.min(ds_N2.wavelength), np.min(ds_O2.wavelength), np.min(ds_CO2.wavelength)])

# transmission_2D: An xarray object with a look-up table the transmission values of the filter per angle of incidence and wavelength
transmission_2D = get_if_transmission(fpath, wave_lims = [wave_min, wave_max])

# ds: Aggregated RR cross sections per ray
# ds_mean: Aggregated RR cross sections averaging over all rays
# ds_central: Aggregated RR cross sections for the central ray (no beam divergence)
# ds_zero_aoi: Aggregated RR cross sections with omega_if = 0
ds, ds_mean, ds_zero_aoi = \
    aggregate_cross_sections_1D(aoi = omega_if.copy(), 
                                transmission = transmission_2D.copy(),
                                ds_N2 = ds_N2, ds_O2 = ds_O2, ds_CO2 = ds_CO2)
    

# ext_err: Raman extinction error per angle and vertical level in Mm-1
ext_err = extinction_error(ds_real = ds_mean, ds_zero_aoi = ds_zero_aoi, hwin = 2)

# bsc_mol: Molecular backscatter coefficient per vertical level in m-1 sr-1
bsc_mol = molecular_bsc_coef(fpath, height, laser_wv, istotal = False)

# bsc_aer: Molecular aerosol coefficient per vertical level in m-1 sr-1
bsc_aer = aerosol_bsc_coef(height,
                           height_nodes = np.array([0., 0.5, 1., 2., 3., 12.]),
                           bsc_nodes = 1e-6 * (355./ laser_wv) * np.array([20., 20., 20.,5.,0.,0.]))

# bsc_err: Raman aerosol backscatter error per vertical level in Mm-1 sr-1
bsc_err = backscatter_error(ds_mean = ds_mean, ds_zero_aoi = ds_zero_aoi, 
                            bsc_aer = bsc_aer, bsc_mol = bsc_mol)

# xcs_err: Backscatter cross section relative error per vertical level in Mm-1 sr-1
xcs_err = cross_section_error(ds_mean = ds_mean, ds_zero_aoi = ds_zero_aoi)

near_far_lim = 1.
nr_tick = 0.1
fr_tick = 1.
ext_lim = np.round(10. * (355. / laser_wv), decimals = 1)
bsc_lim = np.round(0.2 * (355. / laser_wv), decimals = 1)
xcs_lim = np.max(np.abs(xcs_err))
nr_ticks = np.arange(np.round(height[0],decimals = 1) - nr_tick, np.round(height[height <= near_far_lim][-1],decimals = 1) + nr_tick, nr_tick)
fr_ticks = np.arange(np.round(height[0] - nr_tick,decimals = 0), np.round(height[-1],decimals = 0) + fr_tick, fr_tick)

dir_plt = f'./plots/aoi_sims_{pattern}_{use_filter}'
os.makedirs(dir_plt, exist_ok = True)

np.savetxt(os.path.join(dir_plt,f'errors_{laser_wv}.txt'), np.vstack((xcs_err.height.values, ext_err.values, xcs_err.values)).T, header = 'Height Ext_Err Bsc_Err')
np.savetxt(os.path.join(dir_plt,f'filter_{laser_wv}.txt'), np.vstack((transmission_2D.wavelength.values, transmission_2D[0,:].values)).T, header = 'Wavelength Transmission')

# Extinction Error - Near
plt.figure(figsize = (6,4))
ext_err.loc[:near_far_lim].plot(ylim = [-ext_lim,ext_lim])
plt.plot([0.3,0.3],[-ext_lim,ext_lim],'--',color = 'tab:red')
plt.title(f'Near range zoomed - {laser_wv}nm')
plt.ylabel('Extinction Absolute Error [$Mm^{-1}$]')
plt.xlabel('Height [km]')
plt.xticks(ticks = nr_ticks)
plt.axis([nr_ticks[0],nr_ticks[-1],-ext_lim,ext_lim])
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(dir_plt,f'ext_nr_{laser_wv}.png'), dpi = 300)
plt.show()

# Backscatter Error - Near
plt.figure(figsize = (6,4))
bsc_err.loc[:near_far_lim].plot(ylim = [-bsc_lim,bsc_lim])
plt.plot([0.3,0.3],[-bsc_lim,bsc_lim],'--',color = 'tab:red')
plt.title(f'Near range zoomed - {laser_wv}nm')
plt.ylabel('Backscatter Absolute Error [$Mm^{-1} sr^{-1}$]')
plt.xlabel('Height [km]')
plt.xticks(ticks = nr_ticks)
plt.axis([nr_ticks[0],nr_ticks[-1],-bsc_lim,bsc_lim])
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(dir_plt,f'bsc_nr_{laser_wv}.png'), dpi = 300)
plt.show()

# Extinction Error
plt.figure(figsize = (6,4))
ext_err.plot(ylim = [-ext_lim,ext_lim])
plt.title(f'Far range - {laser_wv}nm')
plt.ylabel('Extinction Absolute Error [$Mm^{-1}$]')
plt.xlabel('Height [km]')
plt.xticks(ticks = fr_ticks)
plt.axis([fr_ticks[0],fr_ticks[-1],-ext_lim,ext_lim])
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(dir_plt,f'ext_fr_{laser_wv}.png'), dpi = 300)
plt.show()

# Backscatter Error
plt.figure(figsize = (6,4))
bsc_err.plot(ylim = [-bsc_lim,bsc_lim])
plt.title(f'Far range - {laser_wv}nm')
plt.ylabel('Backscatter Absolute Error [$Mm^{-1} sr^{-1}$]')
plt.xlabel('Height [km]')
plt.xticks(ticks = fr_ticks)
plt.axis([fr_ticks[0],fr_ticks[-1],-bsc_lim,bsc_lim])
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(dir_plt,f'bsc_fr_{laser_wv}.png'), dpi = 300)
plt.show()


# AoI on IF
fig = plt.figure(figsize=(6. , 3.))
ax = fig.add_axes([0.07,0.16,0.83,0.74])
ax.set_title('AoI effect on the IF transmission')
ax.scatter(ds_N2.wavelength,ds_N2[0,:],marker = 'o',s=1)
ax.set_ylabel('N2 RR Bsc Cross-Section [$m^{-1} sr^{-1}$]')
ax.stem(ds_N2[0,:].wavelength,ds_N2[0,:],basefmt=" ")
ax.set_ylim([0,1.5*np.max(ds_N2[0,:])])
ax.set_xlabel('Wavelenght [nm]')
ax2 = plt.twinx()
ax2.plot(transmission_2D.wavelength,transmission_2D[0,:],label = f'{transmission_2D.aoi.values[0]}°',color= 'tab:green')
ax2.plot(transmission_2D.wavelength,transmission_2D[100,:],label = f'{transmission_2D.aoi.values[100]}°',color='tab:green',linestyle='--')
ax2.set_ylim([0,100])
ax2.set_ylabel('IF Transmission (%)')
ax2.set_xlim(np.ceil(ds_N2.wavelength.values[22]),np.floor(laser_wv)+1)
ax2.vlines(laser_wv,0,100,color='tab:purple',linewidth=3)
ax2.legend(title='AoI')
plt.savefig(os.path.join(dir_plt,f'aoi_on_filter_{laser_wv}.png'), dpi = 300)
plt.show()

fig = plt.figure(figsize=(5. , 4.))
ax = fig.add_axes([0.14,0.12,0.80,0.82])
cmesh = ax.pcolormesh(omega_if.rays, omega_if.height, omega_if, cmap ='RdBu_r', vmin = -0.8, vmax=0.8)
ax.set_xticks(np.arange(0,omega_if.rays.size+100,100))
ax.set_ylim(omega_if.height[0],3.75)
ax.set_yticks(np.hstack((omega_if.height[0], np.arange(1.,4.,0.5))))
fig.colorbar(cmesh, ax = ax, label = 'AoI in degrees', extend = 'both')
ax.set_ylabel('Height [km]')
ax.set_xlabel('Rays per Height')
ax.set_title('Simulated AoI on the IF - Zemax')
plt.savefig(os.path.join(dir_plt,f'rays_{laser_wv}.png'), dpi = 300)
plt.show()

# # Near Range
# ds_mean[0,0,:].loc[:near_far_lim].plot(label = 'mean')
# ds_central[0,0,:].loc[:near_far_lim].plot(label = 'central')
# ds_zero_aoi[0,0,:].loc[:near_far_lim].plot(label = 'zero_aoi')
# plt.legend()
# plt.title('Near range zoomed - BD/2: 1.0mrad')
# plt.ylabel('Cross Section')
# plt.xlabel('Height [km]')
# plt.show()

# # Far Range
# ds_mean[0,0,:].loc[near_far_lim:].plot(label = 'mean')
# ds_central[0,0,:].loc[near_far_lim:].plot(label = 'central')
# ds_zero_aoi[0,0,:].loc[near_far_lim:].plot(label = 'zero_aoi')
# plt.legend()
# plt.title('Near range zoomed - BD/2: 1.0mrad')
# plt.ylabel('Cross Section')
# plt.xlabel('Height [km]')
# plt.show()

# # Near Range
# ds_mean[0,0,:].loc[:near_far_lim].plot(label = 'mean')
# ds_central[0,0,:].loc[:near_far_lim].plot(label = 'central')
# ds_zero_aoi[0,0,:].loc[:near_far_lim].plot(label = 'zero_aoi')
# plt.legend()
# plt.title('Near range zoomed - BD/2: 1.0mrad')
# plt.ylabel('Cross Section')
# plt.xlabel('Height [km]')
# plt.show()

# # Far Range
# ds_mean[0,0,:].loc[near_far_lim:].plot(label = 'mean')
# ds_central[0,0,:].loc[near_far_lim:].plot(label = 'central')
# ds_zero_aoi[0,0,:].loc[near_far_lim:].plot(label = 'zero_aoi')
# plt.legend()
# plt.title('Near range zoomed - BD/2: 1.0mrad')
# plt.ylabel('Cross Section')
# plt.xlabel('Height [km]')
# plt.show()

# Near Range
# # Cross section Variation
# xcs_err.loc[:near_far_lim].plot()
# plt.plot([0.3,0.3],[-xcs_lim,xcs_lim],'--',color = 'tab:red')
# plt.title(f'Near range zoomed - {laser_wv}nm')
# plt.ylabel('Bsc Cross Section Variation')
# plt.xlabel('Height [km]')
# plt.xticks(ticks = nr_ticks)
# plt.axis([nr_ticks[0],nr_ticks[-1],-xcs_lim,xcs_lim])
# plt.grid()
# plt.savefig(f'./plots/aoi_sims_{pattern}/nr_{laser_wv}_xcs_err.png', dpi = 300)
# plt.tight_layout()
# plt.show()
# filters = np.empty((aoi.size),dtype = object)

# Far Range
# # Cross section Variation
# xcs_err.plot()
# plt.title(f'Far range - {laser_wv}nm')
# plt.ylabel('Bsc Cross Section Variation')
# plt.xlabel('Height [km]')
# plt.xticks(ticks = fr_ticks)
# plt.axis([fr_ticks[0],fr_ticks[-1],-xcs_lim,xcs_lim])
# plt.grid()
# plt.tight_layout()
# plt.savefig(f'./plots/aoi_sims_{pattern}/fr_{laser_wv}_xcs_err.png', dpi = 300)
# plt.show()

# for j in range(aoi.size):
#     if real_if == True:
#         filters[j] = raman_scattering.CustomFilter(wavelengths = if_wave + wv_shift[j], transmittances = if_tran)    
#     else:
#         filters[j] = raman_scattering.SquareFilter(wavelength = if_cwl + wv_shift[j], width = if_fwhm)  


                    
