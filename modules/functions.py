#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:16:33 2024

@author: nikos
"""

from .constants import hc, k_b, eps_o, h, c
import numpy as np

def wavenumber_shift_to_wavelength(wavelength, delta_nu):
    
    """Calculates the wavelength of a line from its wavenumber shift. 
    
    Parameters: 
    -------------
    wavelength: float
        Initial wavelength [nm]
    
    delta_nu: float
        Line wavenumber shift [m⁻¹].
    
    Returns
    -------
    lamda_nm : float
       Shifted wavelength [nm]
    
    """
    
    lamda_nm = 1 / (1 / wavelength + np.array(delta_nu) * 10 ** -9)
    
    return(lamda_nm)

def rotational_energy(J, B, D):
    """ Calculates the rotational energy of a homonuclear diatomic molecule for
    quantum number J.

    Parameters
    ----------
    J : int or array of ints
       Rotational quantum number.
    B : float
        Specific rotational constant for a rigid rotor molecule at vibrational level v     
        
    D : float
       Centrifugal distortion constant for a rigid rotor molecule at vibrational level v

    Returns
    -------
    E_rot : float
       Rotational energy of the molecule (J)
    """       
    
    E_rot = (B * J * (J + 1) - D * J ** 2 * (J + 1) ** 2) * hc
    
    return E_rot

def vibrational_energy(V, nu_vib):
    """ Calculates the vibrational energy of a homonuclear diatomic molecule for
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
    
    E_vib = hc * nu_vib * (V + 0.5)
    
    return E_vib



def placzek_teller(J, branch):
    """Provides the Placzek_Teller coefficients for a specific Raman branch
    
    Parameters
    ----------
    J : float or 1D array of floats
       Rotational quantum number.

    branch : list
       weights ofr odd and even quantum numbers

    Returns
    -------
    X : float or 1D array of float
       The Placzek_Teller coefficient (probability of scattering at a specific brunch)
    
    """
    
    if branch == 'Q':
        X = (J * (J + 1.)) / ((2.*J - 1.) * (2. * J + 3.)) 
        
    elif branch == 'S':
        X = (3. / 2.) * ((J + 1.) * (J + 2.)) / ((2. * J + 1.) * (2. * J + 3.))   
        
    elif branch == 'O':
        X = (3. / 2.) * (J * (J - 1.)) / ((2. * J - 1.) * (2. * J + 1.))  
        
        if (isinstance(J, float) or isinstance(J, int)) and J < 2:
            X = np.nan
        else:
           X[J < 2] = np.nan 

    return(X)

def statistical_weight(quantum_number, g):
    
    """ 
    Parameters
    ----------
    quantum_number : float or 1D array of float
        The rotational quantum number of the current energy state(s)

    g : list
       weights ofr odd and even quantum numbers

    Returns
    -------
    g_weight : float or 1D array of float
       Statistical weight of a specific energy state
    
    """
    
    g_index = np.remainder(quantum_number, 2)

    g_table = np.array(g)

    g = g_table[(g_index)]
    
    g_weight = g * (2. * quantum_number + 1.)
    
    return(g_weight)

def partition_function_rotational(J, temperature, molecular_parameters, mode = 'rotational_raman'):
    """ Maxwell-Boltzman formula for the partition function 
    (see Apendix of Adam 2009, Eq A3). This method is also applied in Long 2002

    M. Adam, “Notes on temperature-dependent lidar equations,”
    J. Atmos. Ocean. Technol. 26, 1021–1039 (2009).
        
    Parameters
    ----------
    J : float or 1D array of float
        The rotational quantum number of the current energy state(s)

    temperature : float
       Gas temperature in Kelvin
       
    molecular_parameters : dict
       A dictionary containing molecular parameters (specifically, g, B0 and D0).

    V : float
        Vibrational quantum number

    Returns
    -------
    P_rot : float or 1D array of float
       The propability of a molecule being at the current rotational energy state
    
    """

    Js = np.arange(0, 101, 1)
    
    g_weight_partition = statistical_weight(quantum_number = Js, 
                                            g = molecular_parameters['g'])

    g_weight = statistical_weight(quantum_number = J, 
                                  g = molecular_parameters['g'])
    
    if mode == 'rotational_raman':
        B = molecular_parameters['B0']
        D = molecular_parameters['D0']
    elif mode in ['vibrational_raman_N2', 'vibrational_raman_O2']:
        B = molecular_parameters['B1']
        D = molecular_parameters['D1']
    
    E_rot_partition = rotational_energy(Js, B = B, D = D)        
    
    E_rot = rotational_energy(J, B = B, D = D)        
    
    partition_populations = \
        g_weight_partition * np.exp(- E_rot_partition / (k_b * temperature)) 

    state_population = g_weight * np.exp(- E_rot / (k_b * temperature)) 
    
    P_rot = state_population/ np.nansum(partition_populations)

    return(P_rot)

def partition_function_vibrational(V, temperature, molecular_parameters):
    """ Maxwell-Boltzman formula for the partition function 
    (see Apendix of Adam 2009, Eq A3). This method is also applied in Long 2002

    M. Adam, “Notes on temperature-dependent lidar equations,”
    J. Atmos. Ocean. Technol. 26, 1021–1039 (2009).
        
    Parameters
    ----------
    V : float or 1D array of float
        The vibrational quantum number of the current energy state(s)
    
    temperature : float
       Gas temperature in Kelvin
       
    molecular_parameters : dict
       A dictionary containing molecular parameters (specifically, g, B0 and D0).

    Returns
    -------
    P_vib : float or 1D array of float
       The propability of a molecule being at the current vibrational energy state
       
    """
    
    Vs = np.arange(0, 101, 1)

    nu_vib = molecular_parameters['nu_vib']
    
    E_vib_partion = vibrational_energy(Vs, nu_vib = nu_vib)        
    
    E_vib = vibrational_energy(V, nu_vib = nu_vib)        
    
    partition_populations = np.exp(- E_vib_partion / (k_b * temperature))

    state_population = np.exp(- E_vib / (k_b * temperature))
    
    P_vib = state_population / np.nansum(partition_populations)
    
    return(P_vib)

def raman_shift(wavelength, max_J, molecular_parameters, mode, branch):
    """ Calculates the rotational Raman shift (delta en) for the anti-Stokes branch for
    quantum number J.

    Parameters
    ----------
    wavelength: float
        Wavelength before scattering [nm]
        
    max_J: int 
        Maximum rotational quantum number (number of lines considered per branch) 
               
    molecular_parameters : dict
        A dictionary containing molecular parameters.
      
    mode: string
        Choose among: rotational_raman and vibrational_raman
        
        rotational_raman: Corresponds to elastic and pure rotational Raman 
                          lidar channel applications.
        
        vibration_raman: Corresponds to vibrational Raman (V = 1)
                         lidar channel applications.
    
    branch : string
       Select one of Q (central), S (Stokes), O (anti-stokes) 
                         
    Returns
    -------

    lamda_line: float
       Line wavelength [nm]
       
    """

    J = np.arange(0,max_J + 1,1)

    nu = 1E9 / wavelength
    
    if mode == 'rotational_raman':
        B = molecular_parameters['B0']
        D = molecular_parameters['D0']
        nu_offset = 0
        
    elif mode in ['vibrational_raman_N2', 'vibrational_raman_O2']:
        B = molecular_parameters['B1']
        D = molecular_parameters['D1'] 
        nu_offset = molecular_parameters['nu_vib']
    
    if branch == 'O':
        delta_nu_line = B * 2 * (2 * J - 1) - D * (3 * (2 * J - 1) + (2 * J - 1) ** 3)
    
        if (isinstance(J, float) or isinstance(J, int)) and J < 2:
            delta_nu_line = np.nan
        else:
           delta_nu_line[J < 2] = np.nan 

    elif branch == 'Q':
        delta_nu_line = 0. * J

    elif branch == 'S':
        delta_nu_line = -B * 2 * (2 * J + 3) + D * (3 * (2 * J + 3) + (2 * J + 3) ** 3)        
    
    nu_line = nu + delta_nu_line - nu_offset
    
    lamda_line = 1E9 / nu_line

    return lamda_line

def raman_lines(emitted_wavelength, max_J, temperature, molecular_parameters, 
                branch, mode, backscattering = False):
    """ Calculates the rotational Raman backsattering cross section for the Stokes/AntiStokes/Central
    branches for quantum number J at a temperature T.

    Parameters
    ----------
    emitted_wavelength : float
       The emitted wavelength in air [nm]
       
    max_J : float
        Maximum rotational quantum number (number of lines considered per branch) 
       
    temperature : float
       The ambient temperature [K]
       
    molecular_parameters : dict
       A dictionary containing molecular parameters.
       
    branch : string
       Select one of Q (central), S (Stokes), O (anti-stokes) 
       
    mode : string
        Choose among: rotational_raman and vibrational_raman
        rotational_raman: Corresponds to elastic and pure rotational Raman 
                          lidar channel applications.
        vibrational_raman_N2: Corresponds to N2 vibrational Raman (V = 1)
                              lidar channel applications.
        vibrational_raman_O2: Corresponds to O2 vibrational Raman (V = 1)
                              lidar channel applications.
                              
    backscattering : bool
       A scalar. If set to True then the backscattering cross section
       is returned instead of the scattering cross section

    Returns
    -------
    b_s : float
       Backscattering [m^{2}sr^{-1}] or total scattering cross section [m^{2}]
       
    lamda_line: float
       Line wavelength
       
    """

    J = np.arange(0,max_J + 1,1)

    # Wavelengths per line
    lamda_line = \
        raman_shift(emitted_wavelength, 
                    max_J = max_J, 
                    molecular_parameters = molecular_parameters, 
                    mode = mode, 
                    branch = branch)
        
    # Unpack the squared anisotropic polarizability
    gamma_square = molecular_parameters['gamma_square']

    if mode == 'rotational_raman':
        # Probability of being at a rotational energy state J  
        P_rot = \
            partition_function_rotational(J = J, 
                                          temperature = temperature, 
                                          molecular_parameters = molecular_parameters,
                                          mode = mode)
        # Probability of being at the vibrational energy state V = 0
        P_vib = 1.
        
        # Unit conversion factor (no converion for pure rotational Raman)
        b_vk_squared = 1.

        # Anisotropic polarizability
        gamma_square = molecular_parameters['gamma_square']

    elif mode in ['vibrational_raman_N2', 'vibrational_raman_O2']:
        # Probability of being at a rotational energy state J  
        P_rot = \
            partition_function_rotational(J = J, 
                                          temperature = temperature, 
                                          molecular_parameters = molecular_parameters,
                                          mode = mode)
        # Probability of being at the vibrational energy state V = 0
        P_vib = \
            partition_function_vibrational(V = 0, 
                                           temperature = temperature, 
                                           molecular_parameters = molecular_parameters)
        
        # Unit conversion factor
        b_vk_squared = h / (8. * np.pi**2 * c * molecular_parameters['nu_vib'])
        
        # Anisotropic polarizability derivative
        gamma_square = molecular_parameters['gamma_prime_square']
            
    # Placzek-Teller coefficients for each branch: The sum equals unity
    X = placzek_teller(J, branch = branch)

    # Raman Cross sections (applicable for rotational and vibrational lines)
    b_s = (np.pi ** 2 / eps_o ** 2) * np.power(1E-9 * lamda_line, -4) *\
        b_vk_squared * P_rot * P_vib * X * (7. / 45.) * gamma_square
        
    # Converts to the total scattering cross section
    if backscattering == False:
        b_s = b_s * (8. * np.pi / 3.) * (10. / 7.)                                

    return b_s, lamda_line

def xsection_polarized(emitted_wavelength, molecular_parameters, mode = 'rotational_raman', 
                       temperature = 273.15, backscattering = False):
    """ Calculates the backsattering cross section of the isotropically scattered part 
    of the Cabannes line for a molecule or atom (She et al. 2001). The equation
    holds for any kind of molecule (linear, assymetrical etc.)
    
    Parameters
    ----------
    emitted_wavelength : float
       The emmitted wavelength in air [nm]

    molecular_parameters : dict
       A dictionary containing molecular parameters.

    mode : string
        Choose among: rotational_raman and vibrational_raman
        rotational_raman: Corresponds to elastic and pure rotational Raman 
                          lidar channel applications.
        vibrational_raman_N2: Corresponds to N2 vibrational Raman (V = 1)
                              lidar channel applications.
        vibrational_raman_O2: Corresponds to O2 vibrational Raman (V = 1)
                              lidar channel applications.   
                              
    temperature : float
       The ambient temperature [K]
       
    backscattering : bool
       A scalar. If set to True then the total backscattering cross section
       is returned instead of the scattering cross section

    Returns
    -------
    b_s : float or 1D array of floats (same as J)
       Scattering cross section [m^{2}sr^{-1}]
       
    """
        
    if mode == 'rotational_raman':
        # Cabannes wavelength 
        lamda_line = emitted_wavelength
        
        # Probability of being at the vibrational energy state V = 0
        P_vib = 1.
        
        # Unit conversion factor (no converion for pure rotational Raman)
        b_vk_squared = 1.
        
        # Isotropic polarizability
        alpha_square = molecular_parameters['alpha_square']
        
    elif mode in ['vibrational_raman_N2', 'vibrational_raman_O2']:
        # Pure VE line wavelength
        lamda_line = \
            wavenumber_shift_to_wavelength(emitted_wavelength, 
                                           delta_nu = -molecular_parameters['nu_vib'])

        # Probability of being at the vibrational energy state V = 0
        P_vib = \
            partition_function_vibrational(V = 0, 
                                           temperature = temperature, 
                                           molecular_parameters = molecular_parameters)
        
        # Unit conversion factor 
        b_vk_squared = h / (8. *np.pi**2 * c * molecular_parameters['nu_vib'])

        # Isotropic polarizability derivative
        alpha_square = molecular_parameters['alpha_prime_square']
    
    b_s = (np.pi ** 2 / eps_o ** 2) * np.power(1E-9 * lamda_line, -4) * \
        b_vk_squared *P_vib * alpha_square

    # Converts to the total scattering cross section
    if backscattering == False:
        b_s = b_s * (8. * np.pi / 3.)     

    return b_s, lamda_line