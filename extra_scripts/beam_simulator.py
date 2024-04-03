#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 17:11:22 2023

@author: nikos
"""

import numpy as np
from matplotlib import pyplot as plt
from functions.angle_calculator import angle_of_incidence, angle_of_incidence_IF
from functions.rr_calculator import get_astokes_ds, get_if_transmission, aggregate_cross_sections_3D, molecular_bsc_coef
from functions.miscellaneous import aerosol_bsc_coef
# Coordinate system:
# y axis: The line that connects the laser emission and the center of the telescope
# z axis: The vertical axis
# x axis: Vertical to y and y axises so that a Cartecian xyz coordinate system is defined
# laser beam at x = 0, y = 0, z = 0
# telescope center at: x = 0, y = delta, z = 0

# -----------------------------------------------------------------------------
# Input variables
# -----------------------------------------------------------------------------
# laser_wv: The laser wavelength in nm
laser_wv = 1064.

# fpath: Path to the IF ascii file
if np.abs(laser_wv - 355.) < 5:
    fpath = '/home/nikos/Nextcloud/Rotational Raman Lines WG/Data_files/Filters/NBF_Filters_Datasheets/NBF_UV_RR.dat'
if np.abs(laser_wv - 532.) < 5:
    fpath = '/home/nikos/Nextcloud/Rotational Raman Lines WG/Data_files/Filters/NBF_Filters_Datasheets/NBF_VIS_RR.dat'
if np.abs(laser_wv - 1064.) < 5:
    fpath = '/home/nikos/Nextcloud/Rotational Raman Lines WG/Data_files/Filters/NBF_Filters_Datasheets/NBF_IR_RR.dat'

# epsilon: Full beam divergence at 3sigma (99%) in mrad
epsilon = 2.0

# theta_x: Angle between the z axis and the beam spot projection to the zx plane in mrad
theta_x = np.array([0.])

# theta_y: Angle between the z axis and the beam spot projection to the yz plane in mrad
theta_y = np.array([0.])

# delta: Distance between the laser and the center of the telescope (mm)
delta = 212.

# Telescope focal length in mm
f_tel = 1500.

# Collimator focal length in mm
f_col = 80.

# Number of rays
n_rays = 1000

# The height (z axis) in km
height = np.hstack((np.arange(0.05, 1., 0.01),np.arange(1., 2., 0.1),np.arange(2., 12., 0.5)))

# omega: angle of incidence at the telescope apperture (parallel beam) in mrad
omega = angle_of_incidence(theta_x = theta_x, theta_y = theta_y, 
                           height = height, epsilon = epsilon / 3., 
                           delta = delta, n_rays = n_rays)

# omega_if: angle of incidence at the interference filter in degrees
omega_if = angle_of_incidence_IF(omega = omega.copy(), 
                                 f_tel = f_tel, f_col = f_col)

# ds_N2, ds_O2, ds_CO2: The Anti-stokes branch backscattering cross-sections of N2, O2, and CO2 respectively, in m2 sr-1
ds_N2, ds_O2, ds_CO2 = get_astokes_ds(height, laser_wv, istotal = False)

# transmission_2D: An xarray object with a look-up table the transmission values of the filter per angle of incidence and wavelength
transmission_2D = get_if_transmission(fpath)

# ds: Aggregated RR ross sections per ray
# ds_mean: Aggregated RR ross sections averaging over all rays
# ds_central: Aggregated RR ross sections for the central ray (no beam divergence)
# ds_zero_aoi: Aggregated RR ross sections with omega_if = 0
ds, ds_mean, ds_central, ds_zero_aoi = \
    aggregate_cross_sections_3D(aoi = omega_if.copy(), 
                             transmission = transmission_2D.copy(),
                             ds_N2 = ds_N2, ds_O2 = ds_O2, ds_CO2 = ds_CO2)


# ext_err: Raman extinction error per angle and vertical level in Mm-1
ext_err = 1e3 * 0.5 * np.log(ds_zero_aoi/ds_mean).differentiate('height')

# bsc_mol: Molecular backscatter coefficient per vertical level in m-1 sr-1
bsc_mol = molecular_bsc_coef(fpath, height, laser_wv, istotal = False)

# bsc_aer: Molecular aerosol coefficient per vertical level in m-1 sr-1
bsc_aer = aerosol_bsc_coef(height,
                           height_nodes = np.array([0., 0.5, 1., 2., 3., 10.]),
                           bsc_nodes = 1e-6 * (355./ laser_wv) * np.array([20., 20., 20.,5.,0.,0.]))

# bsc_err: Raman aerosol backscatter error per angle and vertical level in Mm-1 sr-1
bsc_err = 1e6 * (1. - ds_mean / ds_zero_aoi) * (bsc_aer + bsc_mol)

xcs_var = ((ds_mean - ds_zero_aoi)/ds_zero_aoi)

near_far_lim = 1.
nr_tick = 0.1
fr_tick = 1.
ext_lim = np.round(10. * (355. / laser_wv), decimals = 1)
bsc_lim = np.round(0.2 * (355. / laser_wv), decimals = 1)
xcs_lim = np.max(np.abs(xcs_var))
nr_ticks = np.arange(np.round(height[0],decimals = 1) - nr_tick, np.round(height[height <= near_far_lim][-1],decimals = 1) + nr_tick, nr_tick)
fr_ticks = np.arange(np.round(height[0] - nr_tick,decimals = 0), np.round(height[-1],decimals = 0) + fr_tick, fr_tick)

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
# Cross section Variation
xcs_var.loc[:near_far_lim].plot()
plt.plot([0.3,0.3],[-xcs_lim,xcs_lim],'--',color = 'tab:red')
plt.title(f'Near range zoomed - {laser_wv}nm')
plt.ylabel('Bsc Cross Section Variation')
plt.xlabel('Height [km]')
plt.xticks(ticks = nr_ticks)
plt.axis([nr_ticks[0],nr_ticks[-1],-xcs_lim,xcs_lim])
plt.grid()
plt.savefig(f'./plots/aoi_sims_ML_DK300_300m-overlap_355-532_2PC-150-170/nr_{laser_wv}_xcs_err.png', dpi = 300)
plt.tight_layout()
plt.show()

# Extinction Error
ext_err.loc[:near_far_lim].plot(ylim = [-ext_lim,ext_lim])
plt.plot([0.3,0.3],[-ext_lim,ext_lim],'--',color = 'tab:red')
plt.title(f'Near range zoomed - {laser_wv}nm')
plt.ylabel('Extinction Absolute Error [$Mm^{-1}$]')
plt.xlabel('Height [km]')
plt.xticks(ticks = nr_ticks)
plt.axis([nr_ticks[0],nr_ticks[-1],-ext_lim,ext_lim])
plt.grid()
plt.tight_layout()
plt.savefig(f'./plots/aoi_sims_ML_DK300_300m-overlap_355-532_2PC-150-170/nr_{laser_wv}_ext_err.png', dpi = 300)
plt.show()

# Backscatter Error
bsc_err.loc[:near_far_lim].plot(ylim = [-bsc_lim,bsc_lim])
plt.plot([0.3,0.3],[-bsc_lim,bsc_lim],'--',color = 'tab:red')
plt.title(f'Near range zoomed - {laser_wv}nm')
plt.ylabel('Backscatter Absolute Error [$Mm^{-1} sr^{-1}$]')
plt.xlabel('Height [km]')
plt.xticks(ticks = nr_ticks)
plt.axis([nr_ticks[0],nr_ticks[-1],-bsc_lim,bsc_lim])
plt.grid()
plt.tight_layout()
plt.savefig(f'./plots/aoi_sims_ML_DK300_300m-overlap_355-532_2PC-150-170/nr_{laser_wv}_bsc_err.png', dpi = 300)
plt.show()

# Far Range
# Cross section Variation
xcs_var.plot()
plt.title(f'Far range - {laser_wv}nm')
plt.ylabel('Bsc Cross Section Variation')
plt.xlabel('Height [km]')
plt.xticks(ticks = fr_ticks)
plt.axis([fr_ticks[0],fr_ticks[-1],-xcs_lim,xcs_lim])
plt.grid()
plt.tight_layout()
plt.savefig(f'./plots/aoi_sims_ML_DK300_300m-overlap_355-532_2PC-150-170/fr_{laser_wv}_xcs_err.png', dpi = 300)
plt.show()

# Extinction Error
ext_err.plot(ylim = [-ext_lim,ext_lim])
plt.title(f'Far range - {laser_wv}nm')
plt.ylabel('Extinction Absolute Error [$Mm^{-1}$]')
plt.xlabel('Height [km]')
plt.xticks(ticks = fr_ticks)
plt.axis([fr_ticks[0],fr_ticks[-1],-ext_lim,ext_lim])
plt.grid()
plt.tight_layout()
plt.savefig(f'./plots/aoi_sims_ML_DK300_300m-overlap_355-532_2PC-150-170/fr_{laser_wv}_ext_err.png', dpi = 300)
plt.show()

# Backscatter Error
bsc_err.plot(ylim = [-bsc_lim,bsc_lim])
plt.title(f'Far range - {laser_wv}nm')
plt.ylabel('Backscatter Absolute Error [$Mm^{-1} sr^{-1}$]')
plt.xlabel('Height [km]')
plt.xticks(ticks = fr_ticks)
plt.axis([fr_ticks[0],fr_ticks[-1],-bsc_lim,bsc_lim])
plt.grid()
plt.tight_layout()
plt.savefig(f'./plots/aoi_sims_ML_DK300_300m-overlap_355-532_2PC-150-170/fr_{laser_wv}_bsc_err.png', dpi = 300)
plt.show()


                    
