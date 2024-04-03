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
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

# -----------------------------------------------------------------------------
# Input variables
# -----------------------------------------------------------------------------
# fpath: Path to the IF ascii file
folder_1 = 'aoi_sims_DK300_300m_355-532_EO85878_cut_355nm_incl_parangles_NBF_UV_RR'
folder_2 = 'aoi_sims_DK300_300m_355-532_EO85878_cut_355nm_incl_parangles_NBF_UV_RR_Theory'
filter_path_1 = f'/home/nikos/Nextcloud/Lidars/Μ-LIDAR/RR_channels/plots/{folder_1}/filter_354.717.txt'
filter_path_2 = f'/home/nikos/Nextcloud/Lidars/Μ-LIDAR/RR_channels/plots/{folder_2}/filter_354.717.txt'

errors_path_1 = f'/home/nikos/Nextcloud/Lidars/Μ-LIDAR/RR_channels/plots/{folder_1}/errors_354.717.txt'
errors_path_2 = f'/home/nikos/Nextcloud/Lidars/Μ-LIDAR/RR_channels/plots/{folder_2}/errors_354.717.txt'

fdata_1 = np.loadtxt(filter_path_1, skiprows = 1)
fdata_2 = np.loadtxt(filter_path_2, skiprows = 1)

edata_1 = np.loadtxt(errors_path_1, skiprows = 1)
edata_2 = np.loadtxt(errors_path_2, skiprows = 1)

dir_plt = '/home/nikos/Nextcloud/Lidars/Μ-LIDAR/RR_channels/plots/comparison_DK300_300m_355-532_EO85878_cut_355nm_incl_parangles'
os.makedirs(dir_plt, exist_ok = True)
 
llim = 353
ulim = 355
tick = 0.5

fig = plt.figure(figsize=(5. , 3.))
ax = fig.add_axes([0.13,0.15,0.81,0.78])
ax.set_title('Filter Comparison')
ax.plot(fdata_1[:,0],fdata_1[:,1], label = 'Real Filter')
ax.plot(fdata_2[:,0],fdata_2[:,1], label = 'Theoretical Filter')
ax.set_xlim(llim,ulim)
ax.set_ylim(0,100)
ax.set_xticks(np.arange(llim, ulim + tick, tick))
plt.minorticks_on()
ax.set_xlabel('Wavelength [nm]')
ax.set_ylabel('Transmission [%]')
ax.grid(which='both')
ax.legend()
plt.savefig(os.path.join(dir_plt,'filters.png'), dpi = 300)
plt.show()

llim = 0
ulim = 10

fig = plt.figure(figsize=(5. , 3.))
ax = fig.add_axes([0.16,0.15,0.81,0.77])
ax.set_title('Near range - 354 nm')
ax.plot(edata_1[:,0],edata_1[:,1], label = 'Real Filter')
ax.plot(edata_2[:,0],edata_2[:,1], label = 'Theoretical Filter')
ax.set_xlim(0.2,1.)
ax.set_ylim(-ulim,ulim)
ax.set_xlabel('Height [km]')
ax.set_ylabel('Extinction Error [$Mm^{-1}$]')
ax.grid()
ax.legend()
ax.vlines(0.3, -ulim, ulim, linestyles = '--', color = 'tab:red')
plt.savefig(os.path.join(dir_plt,'ext_err_near.png'), dpi = 300)
plt.show()

llim = 0
ulim = 1.

fig = plt.figure(figsize=(5. , 3.))
ax = fig.add_axes([0.16,0.15,0.81,0.77])
ax.set_title('Near range - 354 nm')
ax.plot(edata_1[:,0], 100. * edata_1[:,2], label = 'Real Filter')
ax.plot(edata_2[:,0], 100. * edata_2[:,2], label = 'Theoretical Filter')
ax.set_xlim(0.2,1.)
ax.set_ylim(-ulim,ulim)
ax.set_xlabel('Height [km]')
ax.set_ylabel('Relative Backscatter Error [%]')
ax.grid()
ax.legend()
ax.vlines(0.3, -ulim, ulim, linestyles = '--', color = 'tab:red')
plt.savefig(os.path.join(dir_plt,'bsc_err_near.png'), dpi = 300)
plt.show()

llim = 0
ulim = 10

fig = plt.figure(figsize=(5. , 3.))
ax = fig.add_axes([0.16,0.15,0.81,0.77])
ax.set_title('Far range - 354 nm')
ax.plot(edata_1[:,0],edata_1[:,1], label = 'Real Filter')
ax.plot(edata_2[:,0],edata_2[:,1], label = 'Theoretical Filter')
ax.set_xlim(1.,10.)
ax.set_ylim(-ulim,ulim)
ax.set_xlabel('Height [km]')
ax.set_ylabel('Extinction Error [$Mm^{-1}$]')
ax.grid()
ax.legend()
ax.vlines(0.3, -ulim, ulim, linestyles = '--', color = 'tab:red')
plt.savefig(os.path.join(dir_plt,'ext_err_far.png'), dpi = 300)
plt.show()

llim = 0
ulim = 1.

fig = plt.figure(figsize=(5. , 3.))
ax = fig.add_axes([0.16,0.15,0.81,0.77])
ax.set_title('Far range - 354 nm')
ax.plot(edata_1[:,0], 100. * edata_1[:,2], label = 'Real Filter')
ax.plot(edata_2[:,0], 100. * edata_2[:,2], label = 'Theoretical Filter')
ax.set_xlim(1.,10.)
ax.set_ylim(-ulim,ulim)
ax.set_xlabel('Height [km]')
ax.set_ylabel('Relative Backscatter Error [%]')
ax.grid()
ax.legend()
ax.vlines(0.3, -ulim, ulim, linestyles = '--', color = 'tab:red')
plt.savefig(os.path.join(dir_plt,'bsc_err_far.png'), dpi = 300)
plt.show()


                    
