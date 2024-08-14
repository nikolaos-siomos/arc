#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:47:50 2024

@author: nikos
"""

import numpy as np
from .constants import eps_o

def alpha_prime_N2(wavelength):
    
    """ Returns the isotropic polarizability derivative along the molecular 
    bond axis α' of N2 for a given wavelength.
    
    The wavelength dependence of Oddershede et al. 1982 is applied to the 
    measured values at 355 nm from Weitkamp 2005, p. 245
    
    T. Yoshino, H.J. Bernstein, Intensity in the Raman effect: 
    VI. The photoelectrically recorded Raman spectra of some gases,
    Journal of Molecular Spectroscopy, 1958,
    https://doi.org/10.1016/0022-2852(58)90076-6.

    Jens Oddershede, E.Nørby Svendsen, Dynamic polarizabilities and raman 
    intensities of CO, N2, HCl and Cl2, Chemical Physics, 1982,
    https://doi.org/10.1016/0301-0104(82)80004-9.
    

    Parameters
    ----------
    wavelength : float
        wavelength in nm 

    Returns
    -------
    alpha_prime : float
        the isotropic polarizability derivative α' of the gas in C2 J-1 kg-1 (SI)

    """ 

    lamda_nm = np.array([200., 351.1, 363.8, 435.8, 457.9, 488.0, 514.5, 632.8, 2000.])
    
    alpha_prime_lamda = np.array([6.64, 5.23, 5.19, 5.05, 5.02, 4.98, 4.95, 4.88, 4.73])
        
    alpha_prime_weitkamp = np.sqrt(2.62E-14) * (4 * np.pi * eps_o)
    
    norm = alpha_prime_weitkamp / np.interp(355, lamda_nm, alpha_prime_lamda)
    
    alpha_prime = norm * np.interp(wavelength, lamda_nm, alpha_prime_lamda)

    return alpha_prime

def gamma_prime_N2(wavelength):
    
    """ Returns the anisotropic polarizability derivative along the molecular 
    bond axis γ' of N2 for a given wavelength.
    
    The wavelength dependence of Oddershede et al. 1982 is applied to the 
    measured values at 355 nm from Weitkamp 2005
    
    T. Yoshino, H.J. Bernstein, Intensity in the Raman effect: 
    VI. The photoelectrically recorded Raman spectra of some gases,
    Journal of Molecular Spectroscopy, 1958,
    https://doi.org/10.1016/0022-2852(58)90076-6.
    
    Jens Oddershede, E.Nørby Svendsen, Dynamic polarizabilities and raman 
    intensities of CO, N2, HCl and Cl2, Chemical Physics, 1982,
    https://doi.org/10.1016/0301-0104(82)80004-9.
    
    
    Parameters
    ----------
    wavelength : float
        wavelength in nm 
    
    Returns
    -------
    gamm_prime : float
        the anisotropic polarizability derivative γ' of the gas in C2 J-1 kg-1 (SI)
    
    """ 

    lamda_nm = np.array([200., 351.1, 363.8, 435.8, 457.9, 488.0, 514.5, 632.8, 2000.])
    
    gamma_prime_lamda = np.array([8.66, 6.78, 6.74, 6.54, 6.50, 6.45, 6.42, 6.16, 6.12])
        
    gamma_prime_weitkamp = np.sqrt(4.23E-14) * (4 * np.pi * eps_o)

    norm = gamma_prime_weitkamp / np.interp(355., lamda_nm, gamma_prime_lamda)
    
    gamma_prime = norm * np.interp(wavelength, lamda_nm, gamma_prime_lamda)

    return gamma_prime

def alpha_prime_O2():
    
    """ Returns the isotropic polarizability derivative along the molecular 
    bond axis α' of O2 .
    
    The constant measured values at 355 nm from Weitkamp 2005, p. 245 is ussed

    Parameters
    ----------
    wavelength : float
        wavelength in nm 

    Returns
    -------
    alpha_prime : float
        the isotropic polarizability derivative α' of the gas in C2 J-1 kg-1 (SI)

    """ 
    
    alpha_prime_weitkamp = np.sqrt(1.63e-14) * (4 * np.pi * eps_o)
    
    return(alpha_prime_weitkamp)

def gamma_prime_O2():
    
    """ Returns the isotropic polarizability derivative along the molecular 
    bond axis γ' of O2 .
    
    The constant measured values at 355 nm from Weitkamp 2005, p. 245 is ussed

    Parameters
    ----------
    wavelength : float
        wavelength in nm 

    Returns
    -------
    gamma_prime : float
        the isotropic polarizability derivative α' of the gas in C2 J-1 kg-1 (SI)

    """ 
    
    gamma_prime_weitkamp = np.sqrt(6.46e-14) * (4 * np.pi * eps_o)
    
    return(gamma_prime_weitkamp)
