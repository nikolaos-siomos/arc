#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 10:49:14 2024

@author: nikos
"""

import numpy as np

supported_gases = ["N2", "O2", "Ar", "CO2", "H2O"]

def gamma_gas(wavelength, alpha, gas):
    
    """ Returns the anisotropic polarizability parameter γ of N2, O2, Ar, CO2, 
    and H2O for a given wavelength.

    She, C.-Y. Spectral structure of laser light scattering revisited: 
    bandwidths of nonresonant scattering lidars. 
    Appl. Opt. 40, 4875-4884 (2001)

    Parameters
    ----------
    wavelength : float
        wavelength in nm 
    alpha : float
        the isotropic polarizability α of the gas in C2 m2 J-1 (SI)

    gas : string
        must be one of : "N2", "O2", "Ar", "CO2", "H2O"

    Returns
    -------
    gamma : float
        the anisotropic polarizability γ of the gas in C2 m2 J-1 (SI)

    """ 

    if gas not in supported_gases:
        raise  Exception(f'The provided gas ({gas}) is not supported. Please use one of: {supported_gases}')
    
    Fk = kings_factor(wavelength, gas)

    epsilon = (Fk - 1.) * 9 / 2.
    
    gamma = alpha * np.sqrt(epsilon) 

    return gamma


def kings_factor(wavelength, gas):
    
    """ Returns the King's correction factor of N2, O2, and CO2.

    Parameters
    ----------
    wavelength : float
       wavelength in nm
    gas : string
        must be one of : "N2", "O2", "Ar", "CO2", "H2O"

    Returns
    -------
    Fk : float
       Kings factor for the selected gas (unitless)
           
    References
    ----------
    Tomasi, C., Vitale, V., Petkov, B., Lupi, A. & Cacciari, A. Improved
    algorithm for calculations of Rayleigh-scattering optical depth in standard
    atmospheres. Applied Optics 44, 3320 (2005).

    Bates, D. R.: Rayleigh scattering by air, Planetary and Space Science, 32(6),
    785-790, doi:10.1016/0032-0633(84)90102-8, 1984.

    Alms et al. Measurement of the discpersion in polarizability anisotropies. Journal of Chemical Physics (1975)
    """
    
    lamda_um = wavelength * 10 ** -3  # Convert to micrometers, as in the paper

    if gas == 'N2':
        Fk = 1.034 + 3.17e-4 * lamda_um ** -2
    elif gas == 'O2':
        Fk = 1.096 + 1.385e-3 * lamda_um ** -2 + 1.448e-4 * lamda_um ** -4
    elif gas == 'Ar':
        Fk = 1.00
    elif gas == 'CO2':
        Fk = 1.15
    elif gas == 'H2O':
        Fk = 1.001

    return Fk
