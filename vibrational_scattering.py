#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 13:39:54 2024

@author: m.haimerl
"""

''' This module calculates the (vibrational) Raman scattering induced in N2 and O2 molecules in
the atmosphere. It is an application of the method found in:

A. Behrendt and T. Nakamura, "Calculation of the calibration constant of polarization lidar 
and its dependency on atmospheric temperature," Opt. Express, vol. 10, no. 16, pp. 805-817, 2002.
 
The molecular parameters gamma and epsilon are wavelength dependent, and this 
makes the results of the original paper valid only for 532nm. Some new formulas
have been implemented from:

Tomasi, C., Vitale, V., Petkov, B., Lupi, A. & Cacciari, A. Improved 
algorithm for calculations of Rayleigh-scattering optical depth in standard 
atmospheres. Applied Optics 44, 3320 (2005).

and

Chance, K. V. & Spurr, R. J. D. Ring effect studies: Rayleigh scattering, 
including molecular parameters for rotational Raman scattering, and the 
Fraunhofer spectrum. Applied Optics 36, 5224 (1997).

It is not thoroughly tested, so use with care.
'''
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

from .constants import hc, k_b, eps_o, h, c
from .utilities import number_density_at_pt
from .make_gas import N2, O2, Ar, CO2, H2O, relative_concentrations
from .raman_scattering import *
# from .molecular_properties import mol_scatt_props
from .filters import *
import sys

# from arc.constants import hc, k_b, eps_o, h, c
# from arc.utilities import number_density_at_pt
# from arc.make_gas import N2, O2, Ar, CO2, H2O, relative_concentrations
# from arc.raman_scattering import *
# from arc.filters import *
# import sys


def vibrational_energy(V, n_vib):
    """ Calculates the rotational energy of a homonuclear diatomic molecule for
    quantum number J. The molecule is specified by passing a dictionary with
    parameters.

    Parameters
    ----------
    V : int (V = 0, 1, 2, ...)
       Vibrational quantum number.
    nu_vib : int (for N2 = 2331 cm⁻1 -> Adam)
       Vibrational raman shift.

    Returns
    -------
    E_vib : float
       Vibrational energy of the molecule
    """
    E_vib = hc*n_vib*(V + 0.5)
    return E_vib

def vib_rotational_energy(J, molecular_parameters):
    """ Calculates the rotational energy of a homonuclear diatomic molecule for
    quantum number J. The molecule is specified by passing a dictionary with
    parameters.

    Parameters
    ----------
    J : int
       Rotational quantum number.
    molecular_parameters : dict
       A dictionary containing molecular parameters (specifically, B0 and D0).

    Returns
    -------
    E_rot : float
       Rotational energy of the molecule (J)
    """

    B1 = molecular_parameters['B1']
    D0 = molecular_parameters['D0']

    E_rot = (B1 * J * (J + 1) - D0 * J ** 2 * (J + 1) ** 2) * hc
    return E_rot

def vib_wavelength_shift(laser_wave, molecular_parameters):
    """Calculates the wavelength of the center of the Stokes vibrational spectrum 
    according to the first vibrational level. 
    
    Parameters: 
    -------------
    laser_wave: float
    Lidar laser wavelength in [nm].
    
    n_vib: float
    Vibrational wavenumber of the first vibrational level [m⁻¹].
    """
    wavelength = 1/(1/(laser_wave*1E-9) - molecular_parameters['n_vib'])*1E9
    
    return wavelength
    
def vib_raman_shift_stokes(J, molecular_parameters):
    """ Calculates the vibrational rotational Raman shift  (delta_n) for the Stokes branch for
    quantum number J.

    Parameters
    ----------
    J : int 
        Rotational quantum number.
    molecular_parameters: dict 
        Molecular parameters.

    Returns
    -------
    delta_n: float
        Rotational Raman shift [m-1]
    """

    delta_n = raman_shift_stokes(J, molecular_parameters) - molecular_parameters['n_vib']
    return delta_n

def vib_raman_shift_antistokes(J, molecular_parameters):
    """ Calculates the vibrational rotational Raman shift  (delta_n) for the Stokes branch for
    quantum number J.

    Parameters
    ----------
    J : int 
        Rotational quantum number.
    molecular_parameters: dict 
        Molecular parameters.

    Returns
    -------
    delta_n: float
        Rotational Raman shift [m-1]
    """

    delta_n = raman_shift_antistokes(J, molecular_parameters) - molecular_parameters['n_vib']
    return delta_n

def vib_partition_function_summed(molecular_parameters, temperature = 293.15, max_V = 40):
    """ Maxwell-Boltzman formula for the partition function 
    (see Apendix of Adam 2009, Eq A3). This method is also applied in Long 2002

    M. Adam, “Notes on temperature-dependent lidar equations,” 
    J. Atmos. Ocean. Technol. 26, 1021–1039 (2009).
        
    Parameters
    ----------
       
    nu_vib : int (for N2 = 2331 cm⁻1 -> Adam)
       Vibrational raman shift.

    temperature : float
       Gas temperature in Kelvin

    Returns
    -------
    Q_rot : float
       The partition function for J--> infinity
       
    """
    
    Q_vib = np.exp(-(hc*molecular_parameters['n_vib'])/(2*k_b*temperature))* (1/(1-np.exp(-(hc*n_vib)/(k_b*temperature))))
  
    return(Q_vib)
    
def vib_dist_function_summed(molecular_parameters, temperature = 293.15):
    """    
    Parameters
    ----------
       
    nu_vib : int (for N2 = 2331 cm⁻1 -> Adam)
       Vibrational raman shift.

    temperature : float
       Gas temperature in Kelvin

    Returns
    -------
    Q_rot : float
       The partition function for J--> infinity
       
    """
    
    F_MB = 1/(1-np.exp(-(hc*molecular_parameters['n_vib'])/(k_b*temperature)))
  
    return(F_MB)
    
def qm_xsection_vrr_branch(n_incident, J, max_J, temperature, molecular_parameters, branch, alpha_sq = True, istotal = False):
    """ Calculates the vibrational-rotational Raman backsattering cross section for the Stokes/AntiStokes/Central
    branches for quantum number J , vibrational wavenumber n_vib at a temperature T.

    Parameters
    ----------
    n_incident : float
       Wavenumber of incident light [m^-1]
       
    J : int
       Rotational quantum number
       
    max_J : float
       Maximum rotational quantum number (number of lines considered) 
       
    temperature : float
       The ambient temperature [K]
       
    molecular_parameters : dict
       A dictionary containing molecular parameters.
      
    molecule: string
        Desired molecule, e.g. N2 or O2.
        
    branch : string
       Select one of Q (central), S (Stokes), O (anti-stokes) 
       
    istotal : bool
       A scalar. If set to True then the total scattering cross section
       is returned instead of the backscattering cross section

    Returns
    -------
    b_s : float
       Backscattering [m^{2}sr^{-1}] or total scattering cross section [m^{2}]
       
    """

    g_index = np.remainder(int(J), 2)
    g = molecular_parameters['g'][g_index]
    J = int(J)
    PHI = 0.
    
   
    a_square_prime = molecular_parameters['alpha_square_prime']
    gamma_square_prime = molecular_parameters['gamma_square_prime']

    
    # Partion function: the sum of all existing rotational states 
    Q_rot = partition_function_by_summing(max_J = max_J, temperature = temperature, molecular_parameters = molecular_parameters)
    # Q_rot = partition_function_ideal_rigid_rotor(temperature=temperature, molecular_parameters=molecular_parameters)

    # Rotational energy that corresponds to quantum number J for the first vibrational level using the constant B1 instead of B0
    E_factor = np.exp(-vib_rotational_energy(J, molecular_parameters) / (k_b * temperature))
    
    # Vibrational distribution function summed
    vib_dist_func_sum = vib_dist_function_summed(molecular_parameters, temperature)
    
    
    # Placzek-Teller coefficients for each branch: The sum equals unity
    if branch == 'Q':
        n_shifted = n_incident
        if alpha_sq == True:
            PHI = (J*(J + 1.))/((2.*J - 1)*(2.*J + 3.)) + ((45./7.)*a_square_prime/gamma_square_prime)
        else: 
            PHI = (J*(J + 1.))/((2.*J - 1)*(2.*J + 3.))
            
    elif branch == 'S':
        n_shifted = (n_incident + vib_raman_shift_stokes(J, molecular_parameters))
        PHI = (3.*(J + 1.)*(J + 2.))/(2.*(2.*J + 1.)*(2.*J + 3.))
        
    elif branch == 'O':
        n_shifted = (n_incident + vib_raman_shift_antistokes(J, molecular_parameters))

        if J == 0. or J == 1.:
            PHI = 0.
        if J > 1:
            PHI = (3.*J*(J - 1.))/(2.*(2.*J + 1.)*(2.*J - 1.))
    else:
        sys.exit(f'-- Error! Rotational Raman branch type does not exist ({branch} provided)')        

    b_s = (112.*np.pi**4./45.)*n_shifted*((g*(2.*J + 1.)*E_factor)/Q_rot)*(vib_dist_func_sum*h/(8.*np.pi**2.*c*molecular_parameters['n_vib']))*PHI*gamma_square_prime
    
    # Converts to the total scattering cross section
    if istotal:
        b_s = b_s * (8. * np.pi / 3.) * (10. / 7.)                                

    return b_s


class VibrationalRotationalRaman:

    def __init__(self, wavelength, temperature, relative_concentrations, max_J=40, optical_filter = None, istotal = False):
        """
        This class calculates the volume depolarization ratio of the molecular
        backscatter signal detected with a polarization lidar.


        Parameters
        ----------
        wavelength: float
           The lidar emission wavelength (nm)
       temperature: float
           The atmospheric temperature (K)
        relative_concentrations: dictionary with floats
            Relative concentrations of atmospheric gases
        max_J : int
           The number of Raman lines to consider in each branch.

        """
        
        if istotal == True and optical_filter != None:
            print("-- Warning!: A filter transmission function was provided but the 'istotal' parameter was set to True. Please not that the filter won't be taken into account as it is not relevant to the total scattering cross section in lidar applications.")
            optical_filter = None
        
        self.N2_parameters = N2(wavelength, relative_concentrations['N2'])
        self.O2_parameters = O2(wavelength, relative_concentrations['O2'])
        self.Ar_parameters = Ar(wavelength, relative_concentrations['Ar'])
        self.CO2_parameters = CO2(wavelength, relative_concentrations['CO2'])
        self.H2O_parameters = H2O(wavelength, relative_concentrations['H2O'])           
            
        self.optical_filter = optical_filter
        self.temperature = temperature
        self.max_J = max_J
        self.Js = np.arange(0, max_J)
        # self.J_stokes = np.arange(2, max_J)

        self.wavelength = float(wavelength)
        self.wavenumber = 10 ** 9 / self.wavelength # in m-1

        # Calculate the Raman shift for the N2 lines
        self.dn_stokes_N2 = np.array([vib_raman_shift_stokes(J, self.N2_parameters) for J in self.Js])
        self.dn_astokes_N2 = np.array([vib_raman_shift_antistokes(J, self.N2_parameters) for J in self.Js])

        # Convert to wavelegnth
        self.dl_stokes_N2 = 1 / (1 / self.wavelength + np.array(self.dn_stokes_N2) * 10 ** -9)
        self.dl_astokes_N2 = 1 / (1 / self.wavelength + np.array(self.dn_astokes_N2) * 10 ** -9)
        self.dl_Q_N2 = np.zeros([101])
        self.dl_Q_N2[:] = 387.02
        
        # Calculate the Raman shift for the Ο2 lines
        self.dn_stokes_O2 = np.array([vib_raman_shift_stokes(J, self.O2_parameters) for J in self.Js])
        self.dn_astokes_O2 = np.array([vib_raman_shift_antistokes(J, self.O2_parameters) for J in self.Js])

        # Convert to wavelegnth
        self.dl_stokes_O2 = 1 / (1 / self.wavelength + np.array(self.dn_stokes_O2) * 10 ** -9)
        self.dl_astokes_O2 = 1 / (1 / self.wavelength + np.array(self.dn_astokes_O2) * 10 ** -9)
        self.dl_Q_O2 = np.zeros([101])
        self.dl_Q_O2[:] = 375.76
        
        # Calculate the Raman shift for the CΟ2 lines
#        self.dn_stokes_CO2 = np.array([raman_shift_stokes(J, self.CO2_parameters) for J in self.Js])
#        self.dn_astokes_CO2 = np.array([raman_shift_antistokes(J, self.CO2_parameters) for J in self.Js])
#
#        # Convert to wavelegnth
#        self.dl_stokes_CO2 = 1 / (1 / self.wavelength + np.array(self.dn_stokes_CO2) * 10 ** -9)
#        self.dl_astokes_CO2 = 1 / (1 / self.wavelength + np.array(self.dn_astokes_CO2) * 10 ** -9)
        
        self.ds_Q_N2 = np.array([qm_xsection_vrr_branch(
            self.wavenumber, J, max_J, temperature, self.N2_parameters, branch='Q', istotal = istotal) for J in self.Js])
        self.ds_stokes_N2 = np.array([qm_xsection_vrr_branch(
            self.wavenumber, J, max_J, temperature, self.N2_parameters, branch='S', istotal = istotal) for J in self.Js])
        self.ds_astokes_N2 = np.array([qm_xsection_vrr_branch(
            self.wavenumber, J, max_J, temperature, self.N2_parameters, branch='O', istotal = istotal) for J in self.Js])

        self.ds_Q_O2 = np.array([qm_xsection_vrr_branch(
            self.wavenumber, J, max_J, temperature, self.O2_parameters, branch='Q', istotal = istotal) for J in self.Js])
        self.ds_stokes_O2 = np.array([qm_xsection_vrr_branch(
            self.wavenumber, J, max_J, temperature, self.O2_parameters, branch='S', istotal = istotal) for J in self.Js])
        self.ds_astokes_O2 = np.array([qm_xsection_vrr_branch(
            self.wavenumber, J, max_J, temperature, self.O2_parameters, branch='O', istotal = istotal) for J in self.Js])


    def vrr_cross_section(self, pressure = 1013.25, istotal = False):

        """ Caclulate the backscattering cross section for the 
        RR Wings (O and S branches) by summing the lines
    
        Parameters
        ----------
        
        pressure: float
           The atmospheric pressure [hPa]
        
        Returns
        -------
        
        cross_section:
           The backscattering or total scattering cross section of the 
           RR Wings (O and S branches) for a linear molecule or atom. 
           Units are either [m2sr-1] or [m2].  
        
           Cross section type depends on istotal given as input to the
           RotationalRaman class

        cross_sections:
           Same as cross_sections but given separately for each gas
        
        sigma:
           The backscattering or total volumetric 
           scattering coefficient (crossection * number density)
            
        """
        
        optical_filter = self.optical_filter
        
        if optical_filter:
            ds_N2 = (np.nansum(optical_filter(self.dl_stokes_N2) * self.ds_stokes_N2) +\
                        np.nansum(optical_filter(self.dl_astokes_N2) * self.ds_astokes_N2) +\
                             np.nansum(optical_filter(self.dl_Q_N2) * self.ds_Q_N2))
            # print(np.nansum(optical_filter(self.dl_stokes_N2) * self.ds_stokes_N2), np.nansum(optical_filter(self.dl_astokes_N2) * self.ds_astokes_N2), np.nansum(self.ds_Q_N2), ds_N2)
            
            # ds_O2 = (np.nansum(optical_filter(self.dl_stokes_O2) * self.ds_stokes_O2) +\
                        # np.nansum(optical_filter(self.dl_astokes_O2) * self.ds_astokes_O2))
            ds_O2 = 0.
        else:
            ds_N2 = np.nansum(self.ds_stokes_N2) + np.nansum(self.ds_astokes_N2) + np.nansum(self.ds_Q_N2)
            
            # ds_O2 = np.nansum(self.ds_stokes_O2) + np.nansum(self.ds_astokes_O2)
            
            ds_O2 = 0.
            
        ds_CO2 = 0.
        
        ds_Ar = 0.
                
        ds_H2O = 0.
        
        cross_section_rr_gas = np.array([ds_N2, ds_O2, ds_Ar, ds_CO2, ds_H2O])
        
        N = number_density_at_pt(pressure, self.temperature, relative_humidity=0., ideal=True)

        c_N2 = self.N2_parameters['relative_concentration']
        c_O2 = self.O2_parameters['relative_concentration']
        c_Ar = self.Ar_parameters['relative_concentration']
        c_CO2 = self.CO2_parameters['relative_concentration']
        c_H2O = self.H2O_parameters['relative_concentration']
        
        c = np.array([c_N2, c_O2, c_Ar, c_CO2, c_H2O])
  
        gases = {'N2' : '', 'O2' : '', 'Ar' : '', 'CO2' : '', 'H2O' : ''} 

        cross_section_rr = np.nansum(c * cross_section_rr_gas)

        sigma_rr = N * cross_section_rr
    
        cross_sections_rr = dict(zip(gases, cross_section_rr_gas))
                    
        return cross_section_rr, cross_sections_rr, sigma_rr 
    
    
    
    
    def stokes_cross_section(self, pressure = 1013.25, istotal = False):

        """ Caclulate the backscattering cross section for the 
        S branch by summing the lines
    
        Parameters
        ----------
        
        pressure: float
           The atmospheric pressure [hPa]
        
        Returns
        -------
        
        cross_section:
           The backscattering or total scattering cross section of the 
           RR Wings (O and S branches) for a linear molecule or atom. 
           Units are either [m2sr-1] or [m2].  
        
           Cross section type depends on istotal given as input to the
           RotationalRaman class

        cross_sections:
           Same as cross_sections but given separately for each gas
        
        sigma:
           The backscattering or total volumetric 
           scattering coefficient (crossection * number density)
            
        """
        
        optical_filter = self.optical_filter

        if optical_filter:
            ds_N2 = np.nansum(optical_filter(self.dl_stokes_N2) * self.ds_stokes_N2) 
            
            ds_O2 = np.nansum(optical_filter(self.dl_stokes_O2) * self.ds_stokes_O2)
                
            # ds_CO2 = np.nansum(optical_filter(self.dl_stokes_CO2) * self.ds_stokes_CO2)
        else:
            ds_N2 = np.nansum(self.ds_stokes_N2)
            
            ds_O2 = np.nansum(self.ds_stokes_O2)

            # ds_CO2 = np.nansum(self.ds_stokes_CO2)
            
        ds_CO2 = 0.
        
        ds_Ar = 0.
                
        ds_H2O = 0.
        
        cross_section_s_gas = np.array([ds_N2, ds_O2, ds_Ar, ds_CO2, ds_H2O])
        
        N = number_density_at_pt(pressure, self.temperature, relative_humidity=0., ideal=True)

        c_N2 = self.N2_parameters['relative_concentration']
        c_O2 = self.O2_parameters['relative_concentration']
        c_Ar = self.Ar_parameters['relative_concentration']
        c_CO2 = self.CO2_parameters['relative_concentration']
        c_H2O = self.H2O_parameters['relative_concentration']
        
        c = np.array([c_N2, c_O2, c_Ar, c_CO2, c_H2O])
  
        gases = {'N2' : '', 'O2' : '', 'Ar' : '', 'CO2' : '', 'H2O' : ''} 

        cross_section_s = np.nansum(c * cross_section_s_gas)

        sigma_s = N * cross_section_s
    
        cross_sections_s = dict(zip(gases, cross_section_s_gas))
                    
        return cross_section_s, cross_sections_s, sigma_s
    
    
    
    def astokes_cross_section(self, pressure = 1013.25, istotal = False):

        """ Caclulate the backscattering cross section for the 
        O branch by summing the lines
    
        Parameters
        ----------
        
        pressure: float
           The atmospheric pressure [hPa]
        
        Returns
        -------
        
        cross_section:
           The backscattering or total scattering cross section of the 
           O branch for a linear molecule or atom. 
           Units are either [m2sr-1] or [m2].  
        
           Cross section type depends on istotal given as input to the
           RotationalRaman class

        cross_sections:
           Same as cross_sections but given separately for each gas
        
        sigma:
           The backscattering or total volumetric 
           scattering coefficient (crossection * number density)
            
        """
        
        optical_filter = self.optical_filter

        if optical_filter:
            ds_N2 = np.nansum(optical_filter(self.dl_astokes_N2) * self.ds_astokes_N2) 
            
            ds_O2 = np.nansum(optical_filter(self.dl_astokes_O2) * self.ds_astokes_O2)
                
            # ds_CO2 = np.nansum(optical_filter(self.dl_astokes_CO2) * self.ds_astokes_CO2)
        else:
            ds_N2 = np.nansum(self.ds_astokes_N2)
            
            ds_O2 = np.nansum(self.ds_astokes_O2)

            # ds_CO2 = np.nansum(self.ds_astokes_CO2)
            
        ds_CO2 = 0.
        
        ds_Ar = 0.
                
        ds_H2O = 0.
        
        cross_section_o_gas = np.array([ds_N2, ds_O2, ds_Ar, ds_CO2, ds_H2O])
        
        N = number_density_at_pt(pressure, self.temperature, relative_humidity=0., ideal=True)

        c_N2 = self.N2_parameters['relative_concentration']
        c_O2 = self.O2_parameters['relative_concentration']
        c_Ar = self.Ar_parameters['relative_concentration']
        c_CO2 = self.CO2_parameters['relative_concentration']
        c_H2O = self.H2O_parameters['relative_concentration']
        
        c = np.array([c_N2, c_O2, c_Ar, c_CO2, c_H2O])
  
        gases = {'N2' : '', 'O2' : '', 'Ar' : '', 'CO2' : '', 'H2O' : ''} 

        cross_section_o = np.nansum(c * cross_section_o_gas)

        sigma_o = N * cross_section_o
    
        cross_sections_o = dict(zip(gases, cross_section_o_gas))
                    
        return cross_section_o, cross_sections_o, sigma_o

    def Q_cross_section(self, pressure = 1013.25, istotal = False):

        """ Caclulate the backscattering cross section for the 
        Q branch by summing the lines
    
        Parameters
        ----------
        
        pressure: float
           The atmospheric pressure [hPa]
        
        Returns
        -------
        
        cross_section:
           The backscattering or total scattering cross section of the 
           Q branch for a linear molecule or atom. 
           Units are either [m2sr-1] or [m2].  
        
           Cross section type depends on istotal given as input to the
           RotationalRaman class

        cross_sections:
           Same as cross_sections but given separately for each gas
        
        sigma:
           The backscattering or total volumetric 
           scattering coefficient (crossection * number density)
            
        """
        
        optical_filter = self.optical_filter

        if optical_filter:
            ds_N2 = np.nansum(optical_filter(self.dl_Q_N2) * self.ds_Q_N2) 
            
            ds_O2 = np.nansum(optical_filter(self.dl_Q_O2) * self.ds_Q_O2)
                
            # ds_CO2 = np.nansum(optical_filter(self.dl_Q_CO2) * self.ds_Q_CO2)
        else:
            ds_N2 = np.nansum(self.ds_Q_N2)
            
            ds_O2 = np.nansum(self.ds_Q_O2)

            # ds_CO2 = np.nansum(self.ds_Q_CO2)
            
        ds_CO2 = 0.
        
        ds_Ar = 0.
                
        ds_H2O = 0.
        
        cross_section_q_gas = np.array([ds_N2, ds_O2, ds_Ar, ds_CO2, ds_H2O])
        
        N = number_density_at_pt(pressure, self.temperature, relative_humidity=0., ideal=True)

        c_N2 = self.N2_parameters['relative_concentration']
        c_O2 = self.O2_parameters['relative_concentration']
        c_Ar = self.Ar_parameters['relative_concentration']
        c_CO2 = self.CO2_parameters['relative_concentration']
        c_H2O = self.H2O_parameters['relative_concentration']
        
        c = np.array([c_N2, c_O2, c_Ar, c_CO2, c_H2O])
  
        gases = {'N2' : '', 'O2' : '', 'Ar' : '', 'CO2' : '', 'H2O' : ''} 

        cross_section_q = np.nansum(c * cross_section_q_gas)

        sigma_q = N * cross_section_q
    
        cross_sections_q = dict(zip(gases, cross_section_q_gas))
                    
        return cross_section_q, cross_sections_q, sigma_q
    
