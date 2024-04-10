#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:04:48 2023

@author: nick
"""

import sys
sys.path.insert(0, '/project/meteo/work/M.Haimerl/Models')
import matplotlib.ticker
from arc.make_gas import N2, O2
from arc import raman_scattering, vibrational_scattering, utilities
from arc.vibrational_scattering import vib_wavelength_shift
from arc.make_gas import relative_concentrations
import numpy as np
import matplotlib.pyplot as plt
# from utilities import number_density_at_pt

temperature = np.array([293.15])
laser_wv = np.array([354.7])


# generate classes
rrb = raman_scattering.RotationalRaman(wavelength=laser_wv[0], max_J=101, temperature=temperature[0], relative_concentrations = relative_concentrations()) 
vrrb = vibrational_scattering.VibrationalRotationalRaman(wavelength=laser_wv[0], max_J=101, temperature=temperature[0], relative_concentrations = relative_concentrations()) 
IFF = raman_scattering.GaussianFilter(laser_wv, 0.5)   

#######    N2
wave_vib_N2 = vib_wavelength_shift(laser_wv, N2(355.))
dl_N2 = np.hstack([rrb.dl_astokes_N2[2:], laser_wv[0]*np.ones(rrb.ds_Q_N2.size), rrb.dl_stokes_N2])
xsection_N2 = np.hstack([rrb.ds_astokes_N2[2:], rrb.ds_Q_N2, rrb.ds_stokes_N2])
v_dl_N2 = np.hstack([vrrb.dl_astokes_N2[2:], wave_vib_N2*np.ones(vrrb.ds_Q_N2.size), vrrb.dl_stokes_N2])
v_xsection_N2 = np.hstack([vrrb.ds_astokes_N2[2:], vrrb.ds_Q_N2, vrrb.ds_stokes_N2])

filter_waves_N2 = np.linspace(np.min(v_dl_N2), np.max(v_dl_N2), 1000)
IFF_N2 = raman_scattering.GaussianFilter(wave_vib_N2, 0.5)   

#######    O2
wave_vib_O2 = vib_wavelength_shift(laser_wv, O2(355.))
dl_O2 = np.hstack([rrb.dl_astokes_O2[2:], laser_wv[0]*np.ones(rrb.ds_Q_O2.size), rrb.dl_stokes_O2])
xsection_O2 = np.hstack([rrb.ds_astokes_O2[2:], rrb.ds_Q_O2, rrb.ds_stokes_O2])
v_dl_O2 = np.hstack([vrrb.dl_astokes_O2[2:], wave_vib_O2*np.ones(vrrb.ds_Q_O2.size), vrrb.dl_stokes_O2])
v_xsection_O2 = np.hstack([vrrb.ds_astokes_O2[2:], vrrb.ds_Q_O2, vrrb.ds_stokes_O2])

filter_waves_O2 = np.linspace(np.min(v_dl_O2), np.max(v_dl_O2), 1000)
IFF_O2 = raman_scattering.GaussianFilter(wave_vib_O2, 1)   

#%%

###########################'''Plotting'''##################################

fig, ax = plt.subplots(2, figsize=(15,10))     
# aax = fig.add_axes([0.1,0.14,0.80,0.63])

ax[0].scatter(dl_N2, xsection_N2*1E34, color = 'tab:blue', s = 12)
ax[0].bar(dl_N2, xsection_N2*1E34, width = 0.02, color = 'tab:blue', alpha = 0.5, label = 'delta_v = 0')
ax[0].scatter(v_dl_N2, v_xsection_N2*3E35, color = 'tab:red', s = 12)
ax[0].bar(v_dl_N2, v_xsection_N2*3E35, width = 0.02, color = 'tab:red', alpha = 0.5, label = 'delta_v = 1 x30')
#Filters
ax[0].scatter(v_dl_N2, v_xsection_N2*IFF_N2(v_dl_N2)*3E35, color = 'tab:green', s = 12)
ax[0].bar(v_dl_N2, v_xsection_N2*IFF_N2(v_dl_N2)*3E35, width = 0.02, color = 'tab:green', alpha = 0.5, label = 'with filter')
ax[0].plot(filter_waves_N2, IFF_N2(filter_waves_N2),c='tab:olive',label = 'T(λ)')

ax[0].set_ylim(0,5)
ax[0].set_title(f'Depolarized VRR lines of $N_2, \lambda_V = $' + str(float(np.round(wave_vib_N2, 2))) + str('nm'), pad = 10)
# ax[0].set_xlabel('Wavelength [nm]')
ax[0].set_ylabel('$C^{eff,DP}_{bsc,N2,J,k}$ $[m^{2} sr^{-1}]$ x $10^{-34}$')
ax[0].legend(fontsize = 8)
ax[0].minorticks_on()
ax[0].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
ax[0].grid(which = 'both')
# plt.yscale('log')


ax[1].scatter(dl_O2, xsection_O2*1E34, color = 'tab:blue', s = 12)
ax[1].bar(dl_O2, xsection_O2*1E34, width = 0.02, color = 'tab:blue', alpha = 0.5, label = 'delta_v = 0')
ax[1].scatter(v_dl_O2, v_xsection_O2*3E35, color = 'tab:red', s = 12)
ax[1].bar(v_dl_O2, v_xsection_O2*3E35, width = 0.02, color = 'tab:red', alpha = 0.5, label = 'delta_v = 1 x30')
#Filters
ax[1].scatter(v_dl_O2, v_xsection_O2*IFF_O2(v_dl_O2)*3E35, color = 'tab:green', s = 12)
ax[1].bar(v_dl_O2, v_xsection_O2*IFF_O2(v_dl_O2)*3E35, width = 0.02, color = 'tab:green', alpha = 0.5, label = 'with filter')
ax[1].plot(filter_waves_O2, IFF_O2(filter_waves_O2),c='tab:olive',label = 'T(λ)')
        
# ax[1].set_xlim(370,380)
ax[1].set_ylim(0,5)
ax[1].set_title(f'Depolarized VRR lines of $O_2, \lambda_V = $' + str(float(np.round(wave_vib_O2,2))) + str('nm'), pad = 10)
ax[1].set_xlabel('Wavelength [nm]')
ax[1].set_ylabel('$C^{eff,DP}_{bsc,N2,J,k}$ $[m^{2} sr^{-1}]$ x $10^{-34}$')
ax[1].legend(fontsize = 8)
ax[1].minorticks_on()
ax[1].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
ax[1].grid(which = 'both')
plt.show()


#%%

###### Illustrating the effect of IF on the line spectrum

fig3 = plt.figure(figsize=(7,5))     
# axx = fig.add_axes([0.1,0.14,0.80,0.63])

# ax2[0].scatter(dl_N2, xsection_N2*1E34, color = 'tab:blue', s = 12)
# ax2[0].bar(dl_N2, xsection_N2*1E34, width = 0.02, color = 'tab:blue', alpha = 0.5, label = 'delta_v = 0')
plt.scatter(v_dl_N2, v_xsection_N2*1E35, color = 'tab:red', s = 12)
plt.bar(v_dl_N2, v_xsection_N2*1E35, width = 0.02, color = 'tab:red', alpha = 0.5, label = '$N_2$ line spectrum, v=1')
#Filters
plt.scatter(v_dl_N2, v_xsection_N2*IFF_N2(v_dl_N2)*1E35, color = 'tab:green', s = 12)
plt.bar(v_dl_N2, v_xsection_N2*IFF_N2(v_dl_N2)*1E35, width = 0.02, color = 'tab:green', alpha = 0.5, label = 'Filtered line spectrum')
plt.plot(filter_waves_N2, IFF_N2(filter_waves_N2),c='tab:olive',label = 'Gaussian filter, FWHM = 0.5nm')
plt.scatter(filter_waves_N2, IFF_N2(filter_waves_N2 - 0.4),c='tab:olive', alpha = 0.5, marker = '.', label = 'Gaussian filter, FWHM = 0.5nm,\n shifted by 0.4nm')

plt.xlim(383.1,391)
plt.ylim(0,3.)
plt.tick_params(axis= 'x', labelsize=15)
plt.tick_params(axis= 'y', labelsize=15)
plt.title('Filter applied to vibrational rotational Raman lines of $N_2$', pad = 10, fontsize = 15)
plt.xlabel('Wavelength [nm]', fontsize = 15)
plt.ylabel('$\sigma^{diff}_{bsc,N2}$ $[m^{2} sr^{-1}]$ x $10^{-35}$', fontsize = 15)
plt.legend(fontsize =15)
# plt.minorticks_on()
# plt.set_minor_locator(matplotlib.ticker.AutoMinorLocator(1))
plt.grid()
# plt.savefig('/project/meteo/work/M.Haimerl/Models/Plots/' + 'FiltertoVRR_spectraN2.png', dpi = 300)
plt.show()

#%%

###### Overview plots

fig4, ax = plt.subplots(1,2,figsize=(14,7))   
fig4.tight_layout(pad = 0)
# ax = fig.add_axes([0.1,0.14,0.80,0.63])
# ax2 = ax.twinx()

ax[0].scatter(dl_O2, xsection_O2*1E34, color = 'tab:blue', s = 12)
ax[0].bar(dl_O2, xsection_O2*1E34, width = 0.02, color = 'tab:blue', alpha = 0.5)#, label = 'delta_v = 0')
ax[0].scatter(dl_N2, xsection_N2*1E34, color = 'tab:blue', s = 12)
ax[0].bar(dl_N2, xsection_N2*1E34, width = 0.02, color = 'tab:blue', alpha = 0.5, label = '$O_2, N_2$, v = 0')

ax[1].scatter(v_dl_N2[:250], v_xsection_N2[:250]*1E34, color = 'tab:red', s = 12)
ax[1].bar(v_dl_N2[:250], v_xsection_N2[:250]*1E34, width = 0.02, color = 'tab:red', alpha = 0.5, label = '$N_2$, v = 1')
ax[1].scatter(v_dl_O2[:270], v_xsection_O2[:270]*1E34, color = 'tab:green', s = 12)
ax[1].bar(v_dl_O2[:270], v_xsection_O2[:270]*1E34, width = 0.02, color = 'tab:green', alpha = 0.5, label = '$O_2$, v = 1')


###Filters
# ax[1].bar(v_dl_O2, v_xsection_O2*IFF_O2(v_dl_O2)*1E34, width = 0.02, color = 'tab:purple', alpha = 0.5, label = 'With IF')
# ax[1].plot(filter_waves_O2, IFF_O2(filter_waves_O2),c='tab:olive',label = 'Gaussian I, FWHM = 0.5 nm')


###Filters
# ax[1].bar(v_dl_N2, v_xsection_N2*IFF_N2(v_dl_N2)*1E34, width = 0.02, color = 'tab:purple', alpha = 0.5, label = 'With IF')
# ax[1].plot(filter_waves_N2, IFF_N2(filter_waves_N2),c='tab:brown',label = 'Gaussian IF, FWHM = 0.5 nm')

if laser_wv[0] == 532.:
    ax[0].text(528, 10.0, f'v = 0, $\lambda = 355 nm$', fontsize = 15) 
    ax[0].text(565, 9, f'$O_2, v = 1, \lambda_s = 607.30 nm$', fontsize = 15)
    ax[0].text(603, 7.0, f'$N_2, v = 1, \lambda_s = 580.02 nm$', fontsize = 15)
    ax[0].set_ylim(0.00001, 16)
    
else:
    ax[0].text(354., 14.0, f'$\lambda_L = 355$ nm', fontsize = 20) 
    ax[0].text(354.9, 10.0, f'Q', fontsize = 20) 
    ax[0].text(355.6, 8.0, f'S', fontsize = 20)
    ax[0].text(354.2, 8.0, f'O', fontsize = 20) 
    ax[1].text(384.5, 1.5, f'$\lambda_2 = 387.02$ nm', fontsize = 20)
    ax[1].text(373.5, 1.8, f'$\lambda_1 = 375.76$ nm', fontsize = 20)
    ax[0].set_ylim(0.0000001, 16)
    ax[1].set_ylim(0.0000001, 2)
    ax[0].set_xlim(352, 358)
    ax[1].set_xlim(373, 390.5)
    ax[1].yaxis.tick_right()
    
# plt.title(f'Vibrational raman line spectrum for $O_2, N_2$ for a laser wavelength of 354.70 nm', x = 0.0, y = 1.02, fontsize = 20)
plt.title(f'Vibrational raman line spectrum for $O_2, N_2$ for a laser wavelength of 355 nm', x = 0.0, y = 1.02, fontsize = 24)

ax[0].set_xlabel('Wavelength [nm]', fontsize = 20)
ax[1].set_xlabel('Wavelength [nm]', fontsize = 20)
ax[0].set_ylabel('$C^{diff}_{bsc}$ $[m^{2} sr^{-1}]$ x $10^{-34}$', fontsize = 20)
ax[1].set_ylabel('$C^{diff}_{bsc}$ $[m^{2} sr^{-1}]$ x $10^{-34}$', fontsize = 20)
ax[0].tick_params(axis= 'x', labelsize=20)
ax[0].tick_params(axis= 'y', labelsize=20)
ax[1].tick_params(axis= 'x', labelsize=20)
ax[1].tick_params(axis= 'y', labelsize=20)
ax[1].yaxis.set_label_position("right")
ax[1].yaxis.set_label_coords(1.12, 0.5)
ax[0].legend(fontsize = 20)
ax[1].legend(fontsize = 20)
# ax2.legend(bbox_to_anchor=(0.5, 0., 0.5, 0.95), fontsize = 10)
ax[0].minorticks_on()
ax[1].minorticks_on()
ax[0].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
ax[1].xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
ax[0].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
ax[1].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
ax[0].grid()
ax[1].grid()
# ax[0].set_yscale('log')
# ax[1].set_yscale('log')
# plt.savefig('/project/meteo/work/M.Haimerl/Models/Plots/' + 'VRR_spectraN2O2_overview.png', dpi = 300, bbox_inches='tight')
plt.show()

#%%
######### Reproduce Ullas plot from the Raman chapter in the book of Weitcamp

N = utilities.number_density_at_pt(pressure = 1013.25, temperature = temperature[0], relative_humidity=0., ideal=True)
cabanne_pol = rrb.ds_polar_N2 + rrb.ds_polar_O2

#######    N2
wave_vib_N2_new = vib_wavelength_shift(laser_wv, N2(355.))
dl_N2_new = np.hstack([rrb.dl_astokes_N2[2:], laser_wv[0]*np.ones(rrb.ds_Q_N2.size), rrb.dl_stokes_N2])
xsection_N2_new = np.hstack([rrb.ds_astokes_N2[2:], rrb.ds_Q_N2 + cabanne_pol + rrb.ds_Q_O2, rrb.ds_stokes_N2])*relative_concentrations()['N2']*N
v_dl_N2_new = np.hstack([vrrb.dl_astokes_N2[2:], wave_vib_N2*np.ones(vrrb.ds_Q_N2.size), vrrb.dl_stokes_N2])
v_xsection_N2_new = np.hstack([vrrb.ds_astokes_N2[2:], vrrb.ds_Q_N2, vrrb.ds_stokes_N2])*relative_concentrations()['N2']*N

#######    O2
wave_vib_O2_new = vib_wavelength_shift(laser_wv, O2(355.))
dl_O2_new = np.hstack([rrb.dl_astokes_O2[2:], laser_wv[0]*np.ones(rrb.ds_Q_O2.size), rrb.dl_stokes_O2])
xsection_O2_new = np.hstack([rrb.ds_astokes_O2[2:], rrb.ds_Q_O2, rrb.ds_stokes_O2])*relative_concentrations()['O2']*N
v_dl_O2_new = np.hstack([vrrb.dl_astokes_O2[2:], wave_vib_O2*np.ones(vrrb.ds_Q_O2.size), vrrb.dl_stokes_O2])
v_xsection_O2_new = np.hstack([vrrb.ds_astokes_O2[2:], vrrb.ds_Q_O2, vrrb.ds_stokes_O2])*relative_concentrations()['O2']*N


fig5, ax = plt.subplots(1,2,figsize=(14,7))   
fig5.tight_layout(pad = 0)
# ax = fig.add_axes([0.1,0.14,0.80,0.63])
# ax2 = ax.twinx()

ax[0].scatter(dl_O2, xsection_O2_new, color = 'tab:blue', s = 12)
ax[0].bar(dl_O2, xsection_O2_new, width = 0.02, color = 'tab:blue', alpha = 0.5)#, label = 'delta_v = 0')
ax[0].scatter(dl_N2, xsection_N2_new, color = 'tab:blue', s = 12)
ax[0].bar(dl_N2, xsection_N2_new, width = 0.02, color = 'tab:blue', alpha = 0.5, label = '$O_2, N_2$, v = 0')

ax[1].scatter(v_dl_N2[:250], v_xsection_N2_new[:250], color = 'tab:red', s = 12)
ax[1].bar(v_dl_N2[:250], v_xsection_N2_new[:250], width = 0.02, color = 'tab:red', alpha = 0.5, label = '$N_2$, v = 1')
ax[1].scatter(v_dl_O2[:270], v_xsection_O2_new[:270], color = 'tab:green', s = 12)
ax[1].bar(v_dl_O2[:270], v_xsection_O2_new[:270], width = 0.02, color = 'tab:green', alpha = 0.5, label = '$O_2$, v = 1')

if laser_wv[0] == 532.:
    ax[0].text(528, 10.0, f'v = 0, $\lambda = 355 nm$', fontsize = 15) 
    ax[0].text(565, 9, f'$O_2, v = 1, \lambda_s = 607.30 nm$', fontsize = 15)
    ax[0].text(603, 7.0, f'$N_2, v = 1, \lambda_s = 580.02 nm$', fontsize = 15)
    # ax[0].set_ylim(0.00001, 16)
    
else:
    ax[0].text(354., 3E-5, f'$\lambda_L = 355$ nm', fontsize = 20) 
    ax[0].text(354.7, 1E-7, f'Q', fontsize = 20) 
    ax[0].text(355.4, 3E-8, f'S', fontsize = 20)
    ax[0].text(353.8, 3E-8, f'O', fontsize = 20) 
    ax[1].text(384., 1E-8, f'$\lambda_2 = 387.02$ nm', fontsize = 20)
    ax[1].text(372, 1E-8, f'$\lambda_1 = 375.76$ nm', fontsize = 20)
    ax[0].set_ylim(1E-14, 1E-3)
    ax[1].set_ylim(1E-14, 1E-3)
    # ax[1].set_ylim(0.0000001, 2)
    ax[0].set_xlim(350.7, 358.9)
    ax[1].set_xlim(371., 392)
    ax[1].yaxis.tick_right()
    
# plt.title(f'Vibrational raman line spectrum for $O_2, N_2$ for a laser wavelength of 354.70 nm', x = 0.0, y = 1.02, fontsize = 20)
plt.title(f'Raman backscatter line spectrum for $O_2, N_2$ and a laser wavelength of ' +str(laser_wv[0]) + 'nm', x = 0.0, y = 1.02, fontsize = 24)

ax[0].set_xlabel('Wavelength [nm]', fontsize = 20)
ax[1].set_xlabel('Wavelength [nm]', fontsize = 20)
ax[0].set_ylabel('$C^{tot}_{bsc}$ $[m^{2} sr^{-1}]$', fontsize = 20)
ax[1].set_ylabel('$C^{tot}_{bsc}$ $[m^{2} sr^{-1}]$', fontsize = 20)
ax[0].tick_params(axis = 'x', labelsize = 20)
ax[0].tick_params(axis = 'y', labelsize = 20)
ax[1].tick_params(axis = 'x', labelsize = 20)
ax[1].tick_params(axis = 'y', labelsize = 20)
ax[1].yaxis.set_label_position("right")
ax[1].yaxis.set_label_coords(1.15, 0.5)
ax[0].legend(fontsize = 20)
ax[1].legend(fontsize = 20)
# ax2.legend(bbox_to_anchor=(0.5, 0., 0.5, 0.95), fontsize = 10)
ax[0].minorticks_on()
ax[1].minorticks_on()
ax[0].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
ax[1].xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
ax[0].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
ax[1].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
ax[0].grid()
ax[1].grid()
ax[0].set_yscale('log')
ax[1].set_yscale('log')
# plt.savefig('/project/meteo/work/M.Haimerl/Models/Plots/' + 'UllasPlotN2O2_overview.png', dpi = 300, bbox_inches='tight')
plt.show()



