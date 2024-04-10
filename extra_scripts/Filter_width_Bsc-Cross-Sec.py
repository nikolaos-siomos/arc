#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 5 15:57:45 2022

@author: nick
"""

import sys
sys.path.insert(0, '/project/meteo/work/M.Haimerl/Models')
from arc import raman_scattering, vibrational_scattering
from arc.make_gas import relative_concentrations, N2
from arc.vibrational_scattering import vib_wavelength_shift
from arc.utilities import number_density_at_pt
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import us_std
import matplotlib.ticker

####### Initial settings
gas = 'N2'
laser_wv = np.array([354.7])
altitude = np.arange(0.,16000.,1000.)
atm = us_std.Atmosphere()
pressure = np.array([atm.pressure(alt) for alt in altitude]) 
temperature = np.array([atm.temperature(alt) for alt in altitude])

colors = ['tab:red','tab:orange','tab:olive','tab:green','tab:cyan','tab:blue','tab:purple','tab:pink']

# ewp = (1.0007 + 3.46 * 1E-6 * pressure) * 6.1121 * np.exp(17.502 * (temperature - 273.15)/(240.97 + temperature - 273.15))
# c_H2O = ewp / pressure 

# c_N2 = relative_concentrations()['N2']
# c_O2 = relative_concentrations()['O2']
# c_Ar = relative_concentrations()['Ar']
# c_CO2 = relative_concentrations()['CO2']

# c_td = (c_N2 + c_O2 + c_Ar + c_CO2) + 0. * c_H2O
# c_tn = c_td + 0. * c_H2O
# c_tw = c_td + 0. * c_H2O 

# c = {'N2': c_N2 / c_tn, 'O2' : c_O2 / c_tn, 'Ar' : c_Ar / c_tn, 'CO2' : c_CO2 / c_tn, 'H2O' : 0.* c_H2O}

####### Create filters 

FWHM = np.array([0.22, 0.48, 1., 2., 3., 5., 10., 15., np.nan])
f_shape = ['LF','GF', 'Cust_387_0.22', 'Cust_387_0.48', 'Cust_607_0.19', 'Cust_607_0.48']
filter_t = np.array([f'{FWHM[j]}' for j in range(0,len(FWHM))])   #+['Ray']
filters = np.empty((filter_t.size, len(f_shape), laser_wv.size), dtype = object)

filter_wave = []
for jj in laser_wv:
    filter_wave.append(vib_wavelength_shift(jj, N2(jj)))

filter_wavelength = np.asarray(filter_wave)
filter_shift = 0.

if laser_wv[0] == 354.7:
    filter_shift = (vib_wavelength_shift(laser_wv[0], N2(laser_wv[0])) - 387.02)

for i in range(laser_wv.size):    
    for j in range(FWHM.size):
        if FWHM[j] == FWHM[j]:
            filters[j,0,i] = raman_scattering.LorentzianFilter(filter_wavelength[i], FWHM[j])  
            filters[j,1,i] = raman_scattering.GaussianFilter(filter_wavelength[i], FWHM[j])   
            # filters[j,2,i] = raman_scattering.SquareFilter(filter_wavelength[i] + filter_shift,FWHM[j])  

##Custom filters 
if laser_wv[0] == 354.7 or laser_wv[0] == 355.:
    wavelengths387 = np.linspace(filter_wavelength[0] - 3 + filter_shift, filter_wavelength[0] + 3 + filter_shift, 600) 
    filters[0,2,0] = raman_scattering.FileFilter('/project/meteo/work/M.Haimerl/Models/arc/Real_filters/IFF_387_2cav_BW0.22.txt', shift = filter_shift)   
    filters[0,3,0] = raman_scattering.FileFilter('/project/meteo/work/M.Haimerl/Models/arc/Real_filters/IFF_387_2cav_BW0.48.txt', shift = filter_shift)  

if laser_wv[0] == 532.:
    wavelengths607 = np.linspace(filter_wavelength[0] - 3 + filter_shift, filter_wavelength[0] + 3 + filter_shift, 600)
    filters[0,2,0] = raman_scattering.FileFilter('/project/meteo/work/M.Haimerl/Models/arc/Real_filters/IFF_607_2cav_BW0.19.txt')  
    filters[0,3,0] = raman_scattering.FileFilter('/project/meteo/work/M.Haimerl/Models/arc/Real_filters/IFF_607_2cav_BW0.48.txt')              

bsc =  xr.DataArray(dims = ['FWHM', 'shape', 'wavelength', 'altitude'], coords = [filter_t, f_shape, laser_wv, altitude])
bsc_fill = bsc.values.copy()

#%%

for k in range(len(f_shape)):
    for i in range(laser_wv.size):
        for j in range(altitude.size):
                
            # N2 = make_gas.N2(laser_wv[i], relative_concentration = c['N2'][j])
            # O2 = make_gas.O2(laser_wv[i], relative_concentration = c['O2'][j])
            # Ar = make_gas.Ar(laser_wv[i], relative_concentration = c['Ar'][j])
            # CO2 = make_gas.CO2(laser_wv[i], relative_concentration = c['CO2'][j])
            # H2O = make_gas.H2O(laser_wv[i], relative_concentration = c['H2O'][j])
            
            for l in range(len(filter_t)):
                vrrb = vibrational_scattering.VibrationalRotationalRaman(wavelength = laser_wv[i], max_J = 101, temperature=temperature[j], optical_filter = filters[l,k,i], relative_concentrations = relative_concentrations())

                ds_flt_lkij = vrrb.vrr_cross_section()[0]
                bsc_fill[l,k,i,j] = ds_flt_lkij

bsc.values = bsc_fill

#%%
#### Plot custom filters

fig = plt.figure(figsize = (7,5))
# fig.tight_layout(pad = 10)
ax = fig.add_axes([0.1,0.17,0.85,0.70])

if laser_wv[0] == 354.7 or laser_wv[0] == 355.:
    ax.plot(wavelengths387, filters[0,2,0].transmission_function(wavelengths387), label = 'Double cavity IF, FWHM = 0.22nm, $\lambda_c = 387.02$nm', color = 'C0') 
    ax.plot(wavelengths387, filters[0,3,0].transmission_function(wavelengths387), label = 'Double cavity IF, FWHM = 0.48nm, $\lambda_c = 387.02$nm', color = 'C1')

if laser_wv[0] == 532.:
    ax.plot(wavelengths607, filters[0,2,0].transmission_function(wavelengths607), label = 'Double cavity IF, FWHM = 0.22nm, $\lambda_c = 607.76$nm', color = 'C0') 
    ax.plot(wavelengths607, filters[0,3,0].transmission_function(wavelengths607), label = 'Double cavity IF, FWHM = 0.48nm, $\lambda_c = 607.76$nm', color = 'C1')

# ax.set_xlim(386.2, 387.8)
ax.set_ylim(0.0001,1.05)
ax.tick_params(axis= 'x', labelsize=15)
ax.tick_params(axis= 'y', labelsize=15)
ax.set_xlabel('Wavelength [nm]', fontsize = 15)
ax.set_ylabel('Transmission', fontsize = 15)
ax.grid()
ax.legend(bbox_to_anchor=(0.5, 0., 0.52, 1.3), fontsize = 15)
# plt.savefig('/project/meteo/work/M.Haimerl/Models/Plots/' + 'Custom_filters_387.png', dpi = 300, bbox_inches='tight')
plt.show()

#%%

def relative_deveation(array):
    return (array.max() - array.min())/(array.min())*100

deveations_custom387 = []
deveations_custom607 = []
deveations_gau = []
deveations_lr = []


for i in range(laser_wv.size):
    for l in range(len(FWHM)-1):
        deveations_gau.append(relative_deveation(bsc[l,0,i,:].values))
        deveations_lr.append(relative_deveation(bsc[l,1,i,:].values))
    
if laser_wv[0] == 354.7 or laser_wv[0] == 355.:
    dbsc_cust387_022 = (bsc[0,2,0,:].copy())
    dbsc_cust387_048 = (bsc[0,3,0,:].copy())
    deveations_custom387.append(relative_deveation(dbsc_cust387_022.values))
    deveations_custom387.append(relative_deveation(dbsc_cust387_048.values))
    
if laser_wv[i] == 532.:
    dbsc_cust607_019 = (bsc[0,2,0,:].copy())
    dbsc_cust607_048 = (bsc[0,3,0,:].copy())
    deveations_custom607.append(relative_deveation(dbsc_cust607_019.values))
    deveations_custom607.append(relative_deveation(dbsc_cust607_048.values))


for i in range(laser_wv.size):
    fig = plt.figure(figsize=(7,3))     
    ax = fig.add_axes([0.1,0.17,0.85,0.70])
    
    for l in range(len(FWHM)-1):
        dbsc_lr = (bsc[l,0,i,:].copy())
        dbsc_gau = (bsc[l,1,i,:].copy())

        ax.plot(np.nan*dbsc_lr, altitude/1000.,  color = colors[l], label = str(FWHM[l]))
        ax.plot(dbsc_lr*1E33, altitude/1000., '--', color = colors[l], linewidth = 2)
        ax.plot(dbsc_gau*1E33, altitude/1000., color = colors[l], linewidth = 2)
        
    if laser_wv[i] == 354.7 or laser_wv[i] == 355.:
        ax.scatter(dbsc_cust387_022*1E33, altitude/1000., color = 'brown', marker = '.', linewidth = 2, label = '0.22, cust. IF')
        ax.scatter(dbsc_cust387_048*1E33, altitude/1000., color = 'purple', marker = '.', linewidth = 2, label = '0.48, cust. IF')

    if laser_wv[i] == 532.:
        ax.scatter(dbsc_cust607_019*1E33, altitude/1000., color = 'C0', marker = '.', linewidth = 2, label = '0.19, cust. IF')
        ax.scatter(dbsc_cust607_048*1E33, altitude/1000., color = 'C6', marker = '.', linewidth = 2, label = '0.48, cust. IF')

    # ax.set_xlim([0.8, 1.1])
    ax.set_ylim([0,15.])
    ax.set_yticks(np.arange(0.,16,2.),labels = np.arange(0.,16,2.))
    ax.minorticks_on()
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
    ax.grid(visible = True, which = 'both')
    ax.set_title('Effective backscatter cross-sections for N$_2$ spectrum filtered at $\lambda_c =$' + str(np.round(vib_wavelength_shift(laser_wv[0], N2(laser_wv[0])),2)) +'nm', pad = 10, loc = 'left')
    ax.set_xlabel('$\sigma_{bsc,m}^{eff}$ ')
    ax.set_xlabel('$\sigma_{bsc,N_2}^{eff}$ $\cdot 10^{33}$')
    ax.set_ylabel('Altitude [km]')
    ax.legend(title = 'FWHM [nm]', bbox_to_anchor=(0.5, 0., 0.79, 1.03))
    # plt.xscale('log')
    
    # if laser_wv[i] == 355:
        # fig.savefig('/project/meteo/work/M.Haimerl/Models/Plots/' + 'Bsc_variation_355.png', dpi = 300, bbox_inches='tight')
    # else:
        # fig.savefig('/project/meteo/work/M.Haimerl/Models/Plots/' + 'Bsc_variation_607png', dpi = 300, bbox_inches='tight')
    
    plt.show()
    
#%%

fig3 = plt.figure()
plt.scatter(FWHM[:-1], deveations_gau, label = 'Gaussian IF', color = 'black')
plt.scatter(FWHM[:-1], deveations_lr, label = 'Lorenzian IF', color = 'brown')
plt.xlabel('FWHM [nm]', fontsize = 15)
plt.ylabel('Relative deviation [%]', fontsize = 15)
plt.xlim(0, 16)
# plt.ylim(0, 1.1)
plt.grid()

if laser_wv[0] == 354.7 or laser_wv[0] == 355.:
    plt.scatter([0.22, 0.48], deveations_custom387, label = 'Custom IF', color = 'blue')
    plt.title('Relavtive deviation of backscatter cross-section\n for IF centered at $\lambda_c$ = 387.02 nm', fontsize =15)
    # plt.title('Relavtive deviation of backscatter cross-section\n for IF centered at $\lambda_c$ = 386.67', fontsize =15)
    plt.legend()
    # plt.savefig('/project/meteo/work/M.Haimerl/Models/Plots/' + 'Bsc_deveation_355.png', dpi = 300, bbox_inches='tight')
if laser_wv[0] == 532.:
    plt.scatter([0.19, 0.48], deveations_custom607, label = 'Custom IF')
    plt.title('Relavtive deviations for laser wavelength of $\lambda_L$ = 532nm')
    plt.legend()
    # plt.savefig('/project/meteo/work/M.Haimerl/Models/Plots/' + 'Bsc_deveation_607.png', dpi = 300, bbox_inches='tight')

plt.show()
#%%
# '''deveations for different shifts'''

deveation_gau_0 = np.array([0.0005250830431221622,
  0.0024198288495314115,
  0.007327250584895541,
  0.010241010654391407,
  0.008135426197434466,
  0.0042859140564477025,
  0.001295852021186609,
  0.0006019842525670276])*100

deveation_gau_01 = np.array([0.0016047360851510679,
  0.003203696329522235,
  0.008009027715247015,
  0.010589677013954857,
  0.008282940882207069,
  0.0043179310217566665,
  0.0012984668425247035,
  0.0006025388127178309])*100

deveation_gau_02 = np.array([0.015050527564944639,
  0.00587993586302578,
  0.009219560179893773,
  0.010911074362781365,
  0.00838314364841834,
  0.004335286534438431,
  0.0012997427912843495,
  0.0006028091689450992] )*100
    
deveation_gau_03 = np.array([0.1510829176795579,
  0.01329146896144771,
  0.011085206434173026,
  0.011201010951592464,
  0.008434502794771499,
  0.004337836419307268,
  0.0012996767862545376,
  0.0006027950340614828])*100
    
deveation_gau_04 =  np.array([0.22955415419964498,
  0.03350729387839228,
  0.013779568867452813,
  0.011450002111321732,
  0.0084347107162382,
  0.00432540702160846,
  0.001298265576058056,
  0.0006024961137150183])*100

deveation_lr_0 =  np.array([0.0012912139985866073,
  0.003484108102638073,
  0.006646379285535294,
  0.008013203312777582,
  0.0070023793132274895,
  0.0045072269315188625,
  0.0016681048276937012,
  0.0008191825232360245])*100

deveation_lr_01 = np.array([0.0029372067315657936,
  0.004559314530259712,
  0.007303115703344373,
  0.008345997593997139,
  0.0071843529534618275,
  0.004569119898710697,
  0.0016760917901856276,
  0.0008211034510290224])*100

deveation_lr_02 = np.array([0.009551208356759315,
  0.007613546238784996,
  0.008608614078669767,
  0.008744281571665026,
  0.007338214688026421,
  0.004603820011591403,
  0.0016791826753974383,
  0.0008217782326861526])*100
    
deveation_lr_03 = np.array([0.021739205223227788,
  0.013039501087720767,
  0.010551119729268678,
  0.009190201181964517,
  0.007457764772566551,
  0.004610421411006969,
  0.0016773472567308332,
  0.0008212036689130387])*100
    
deveation_lr_04 =  np.array([0.042507921256371164,
  0.02071229819072318,
  0.013009067581329829,
  0.009648379424498088,
  0.007533242255205049,
  0.004587974630319752,
  0.0016705787547913339,
  0.0008193802150837427])*100

deveation_cost_0 = [0.04410874469128484, 0.24374457881236722]
deveation_cost_01 = [0.17846471630933244, 0.2728762115995678]
deveation_cost_02 = [2.3995560129272913, 0.5330366332596594]
deveation_cost_03 = [11.001392273953973, 1.8381013361659557]
deveation_cost_04 = [21.712527858887956, 5.212045724326083]

fig = plt.figure(figsize = (7,5))
plt.plot(FWHM[:-1], deveation_gau_0, '.-', markersize = 10, label = 'Gaussian, shift = 0.0')
plt.plot(FWHM[:-1], deveation_gau_01, '.-', markersize = 10, label = 'Gaussian, shift = 0.1')
plt.plot(FWHM[:-1], deveation_gau_02, '.-', markersize = 10, label = 'Gaussian, shift = 0.2')
plt.plot(FWHM[:-1], deveation_gau_03, '.-', markersize = 10, label = 'Gaussian, shift = 0.3')
plt.plot(FWHM[:-1], deveation_gau_04, '.-', markersize = 10, label = 'Gaussian, shift = 0.4')

# plt.plot(FWHM[:-1], deveation_lr_0, '-', label = 'Lorenzian, shift = 0.0')
# plt.plot(FWHM[:-1], deveation_lr_01, '-', label = 'Lorenzian, shift = 0.1')
# plt.plot(FWHM[:-1], deveation_lr_02, '-', label = 'Lorenzian, shift = 0.2')
# plt.plot(FWHM[:-1], deveation_lr_03, '-', label = 'Lorenzian, shift = 0.3')
# plt.plot(FWHM[:-1], deveation_lr_04, '-', label = 'Lorenzian, shift = 0.4')

plt.scatter([0.22, 0.48], deveation_cost_0,  linewidth = 4, label = 'Custom IF, shift = 0.0')
plt.scatter([0.22, 0.48], deveation_cost_01, linewidth = 4, label = 'Custom IF, shift = 0.1')
plt.scatter([0.22, 0.48], deveation_cost_02, linewidth = 4, label = 'Custom IF, shift = 0.2')
plt.scatter([0.22, 0.48], deveation_cost_03, linewidth = 4, label = 'Custom IF, shift = 0.3')
plt.scatter([0.22, 0.48], deveation_cost_04, linewidth = 4, label = 'Custom IF, shift = 0.4')

plt.xlabel('FWHM [nm]', fontsize = 15)
plt.ylabel('Relative deveation [%]', fontsize = 15)
plt.tick_params(axis= 'x', labelsize = 15)
plt.tick_params(axis= 'y', labelsize = 15)
plt.xlim(0, 15.5)
plt.ylim(0.05, 25)
plt.yscale('log')
plt.grid()
plt.title('Relative deveations for different shifts \n of the filter center wavelength', fontsize = 15)
plt.legend(fontsize = 12)
# plt.savefig('/project/meteo/work/M.Haimerl/Models/Plots/' + 'Bsc_deveation_with_shift.png', dpi = 300, bbox_inches='tight')
plt.show()