#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 15:49:30 2022

@author: nick
"""

from .constants import eps_o
from .isotropic_polarizability import alpha_gas
from .anisotropic_polarizability import gamma_gas
from .polarizability_derivatives import alpha_prime_N2, gamma_prime_N2
from .polarizability_derivatives import alpha_prime_O2, gamma_prime_O2
import numpy as np

""" The functions below provide a dictionary with the Raman parameters per molecule 
 
Parameters
----------
wavelength : float
   Wavelength in vacuum [nm]

Returns
-------
parameters : dictionary
   A dictionary with the Raman parameters per molecule
   
"""

def N2(wavelength):

    """ Raman parameters of Nitrogen (N2) """     
    
    alpha = alpha_gas(wavelength = wavelength, gas = 'N2')

    gamma = gamma_gas(wavelength = wavelength, alpha = alpha, gas = 'N2')

    alpha_prime = alpha_prime_N2(wavelength = wavelength)

    gamma_prime = gamma_prime_N2(wavelength = wavelength)

    parameters = {'name': "N_{2}",
                  'molecule_type': 'linear',
                  'B0': 1.989570 * 1E2, # from Weitkamp 2005, in m-1
                  'D0': 5.76 * 1E-4, # from Weitkamp 2005, in m-1
                  'B1': 1.97219 * 1E2, # from Weitkamp 2005, in m-1
                  'D1': 5.76 * 1E-4, # assumed to be the same as D0 because it's not provided in Weitkamp 2005, in m-1
                  'I': 1, # from Behrend 2002
                  'alpha_square': alpha**2,   
                  'gamma_square': gamma**2,  
                  'alpha_prime_square': alpha_prime**2, 
                  'gamma_prime_square': gamma_prime**2,
                  'g': [6, 3], # from Behrend 2002 for even/odd J - alowed molecular spin values depending on J
                  'nu_vib': 233070.} # from Weitkamp 2005, p. 251, in m-1

    return(parameters)

def O2(wavelength):

    """ Raman parameters of Oxygen (O2) """     
        
    alpha = alpha_gas(wavelength = wavelength, gas = 'O2')

    gamma = gamma_gas(wavelength = wavelength, alpha = alpha, gas = 'O2')

    alpha_prime = alpha_prime_O2()

    gamma_prime = gamma_prime_O2()

    parameters = {'name': "O_{2}",
                  'molecule_type': 'linear',
                  'B0': 1.43768 * 1E2, # from Weitkamp 2005, in m-1
                  'D0': 4.85 * 1E-4, # from Weitkamp 2005, in m-1
                  'B1': 1.42188 * 1E2, # from Weitkamp 2005, in m-1
                  'D1': 4.85 * 1E-4, # assumed to be the same as D0 because it's not provided in Weitkamp 2005, in m-1
                  'I': 0, # from Behrend 2002
                  'alpha_square':  alpha**2,#alpha**2, #2.66E-60,
                  'gamma_square':  gamma**2 ,#gamma**2,  #1.26E-60, 
                  'alpha_prime_square': alpha_prime**2, # from Weitkamp 2005, p. 245
                  'gamma_prime_square': gamma_prime**2, # from Weitkamp 2005, p. 245
                  'g': [0, 1], # from Behrend 2002 for even/odd J
                  'nu_vib': 155640.} # from Weitkamp 2005, p. 251, in m-1
    
    return(parameters)

def Ar(wavelength):
    
    """ Raman parameters of Argon (Ar) """     
    
    alpha = alpha_gas(wavelength = wavelength, gas = 'Ar')

    gamma = gamma_gas(wavelength = wavelength, alpha = alpha, gas = 'Ar')


    parameters = {'name': "Ar",
                  'molecule_type': 'spherical',
                  'alpha_square': alpha**2,  
                  'gamma_square': gamma**2} 
    
    return(parameters)


def CO2(wavelength):
        
    """ Raman parameters of carbon dioxide (CO2) """     
    
    alpha = alpha_gas(wavelength = wavelength, gas = 'CO2')

    gamma = gamma_gas(wavelength = wavelength, alpha = alpha, gas = 'CO2')

    parameters = {'name': "CO_{2}",
                  'molecule_type': 'linear',
                  'B0': 0.39027 * 1E2, #from Sioris 2001 PhD thesis, in m-1
                  'D0': 1.27E-7 * 1E2, #from Sioris 2001 PhD thesis, in m-1
                  'I': 0, #from Sioris 2001 PhD thesis
                  'alpha_square': alpha**2,
                  'gamma_square': gamma**2,
                  'g': [1, 0]} #from Sioris 2001 PhD thesis for even/odd J
    
    return(parameters)


def H2O(wavelength):
    
    """ Raman parameters of water vapor (H2O) """     

    alpha = alpha_gas(wavelength = wavelength, gas = 'H2O')

    gamma = gamma_gas(wavelength = wavelength, alpha = alpha, gas = 'H2O')

    parameters = {'name': "H_{2}O",
                  'molecule_type': 'asymetric',
                  'alpha_square': alpha**2,  
                  'gamma_square': gamma**2, }  
    
    return(parameters)