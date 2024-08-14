""" This file includes functions to calculate the refractive index of air
according to Ciddor (1996, 2002), summarized by Tomasi et al. (2005).
"""

import numpy as np
from scipy.interpolate import interp1d
from .constants import eps_o, k_b

valid_methods = {"N2" : ["griesmann_and_burnett_N2", "boerzsoenyi_N2", "peck_and_khanna_N2"],
                 "O2" : ["smith_O2", "zhang_O2"],
                 "Ar" : ["bideau_mehu_larsen_Ar", "boerzsoenyi_Ar", "peck_and_fisher_Ar"],
                 "CO2": ["bideau_mehu_larsen_CO2", "old_CO2"],
                 "H2O": ["cidor_H2O"]}

lower_limit = {"griesmann_and_burnett_N2" :  145., 
               "boerzsoenyi_N2"           :  270., 
               "peck_and_khanna_N2"       : 1000.,
               "smith_O2"                 :  185., 
               "zhang_O2"                 :  288.,
               "bideau_mehu_larsen_Ar"    :  141., 
               "boerzsoenyi_Ar"           :  400.,
               "peck_and_fisher_Ar"       :  468.,
               "bideau_mehu_larsen_CO2"   :  181.,
               "old_CO2"                  :  481., 
               "cidor_H2O"                :  350.}    

upper_limit = {"griesmann_and_burnett_N2" :  270., 
               "boerzsoenyi_N2"           : 1000., 
               "peck_and_khanna_N2"       : 2058.,
               "smith_O2"                 :  288., 
               "zhang_O2"                 : 1800.,
               "bideau_mehu_larsen_Ar"    :  400., 
               "boerzsoenyi_Ar"           : 1000.,
               "peck_and_fisher_Ar"       : 2058.,
               "bideau_mehu_larsen_CO2"   : 1694.,
               "old_CO2"                  : 1817., 
               "cidor_H2O"                : 1200.}

supported_gases = ["N2", "O2", "Ar", "CO2", "H2O"]


"""
The functions below correspond to different methods for the calculation of
the isotropic polarizability per gas in different wavelength ranges:

Parameters
----------
x: float
   Wavelength in vacuum [nm]
   
   
Returns
-------

alpha : float
   Isotropic polarizability 

"""

def griesmann_and_burnett_N2(x):
    """
    griesmann_burnett_N2:
        
        Original formula corresponds to T = 273.15K (0C), e = 1013.25 hPa
        
        Valid in the range 145-270nm
        
        U. Griesmann and J. H. Burnett. Refractivity of nitrogen gas in the 
        vacuum ultraviolet, Opt. lett. 24, 1699-1701 (1999)

    """
    
    x = x * 1E-3  # the original formula is expressed in μm

    e_o = 101325
    T_o = 273.15

    N_o = e_o / (k_b * T_o)  # Number density in the measurement conditions of the RI

    c1 = 1.9662731
    c2 = 22086.66
    c3 = 2.7450825E-2
    c4 = 133.85688

    n_o = 1. + c1 / (c2 - np.power(x, -2)) + c3 / (c4 - np.power(x, -2))

    alpha = (3. * eps_o) * (1. / N_o) * (np.power(n_o, 2) - 1.) / (np.power(n_o, 2) + 2.)

    
    return alpha


def boerzsoenyi_N2(x):
    """
    boerzsoenyi_N2:
        
        Original formula corresponds to T = 273.15K (0C), e = 1000.00 hPa (1000 mbar)
        
        Valid in the range 400-1000nm
                  
        A. Börzsönyi, Z. Heiner, M. P. Kalashnikov, A. P. Kovács, 
        and K. Osvay, Dispersion measurement of inert gases and gas 
        mixtures at 800 nm, Appl. Opt. 47, 4856-4863 (2008)

    """

    x = x * 1E-3  # the original formula is expressed in μm

    e_o = 100000
    T_o = 273.15

    N_o = e_o / (k_b * T_o)  # Number density in the measurement conditions of the RI

    c1 = 39209.95E-8
    c2 = 1146.24E-6
    c3 = 18806.48E-8
    c4 = 13.476E-3

    n_o = np.sqrt(1. + c1 * np.power(x, 2) / (np.power(x, 2) - c2) + c3 * np.power(x, 2) / (np.power(x, 2) - c4))

    alpha = (3. * eps_o) * (1. / N_o) * (np.power(n_o, 2) - 1.) / (np.power(n_o, 2) + 2.)

    return alpha


def peck_and_khanna_N2(x):
    """
    peck_khanna_N2:
        
        Original formula corresponds to T = 273.15K (0C), e = 1013.25 hPa
        
        Valid in the range 468-2058nm
        
        E. R. Peck and B. N. Khanna. Dispersion of nitrogen, 
        J. Opt. Soc. Am. 56, 1059-1063 (1966)

    """

    x = x * 1E-3  # the original formula is expressed in μm

    e_o = 101325
    T_o = 273.15

    N_o = e_o / (k_b * T_o)  # Number density in the measurement conditions of the RI

    c1 = 6.8552E-5
    c2 = 3.243157E-2
    c3 = 144.
    n_o = 1. + c1 + c2 / (c3 - np.power(x, -2))

    alpha = (3. * eps_o) * (1. / N_o) * (np.power(n_o, 2) - 1.) / (np.power(n_o, 2) + 2.)

    return alpha

def smith_O2(x):
    """
    
    smith_O2:
        
        Original formula corresponds to T = 273.15K (0C), e = 1013.25 hPa
        
        Valid in the range 185-288nm
        
        P. L. Smith, M. C. E. Huber, W. H. Parkinson. Refractivities of 
        H2, He, O2, CO, and Kr for 168≤λ≤288 nm,
        Phys Rev. A 13, 199-203 (1976)

    """

    e_o = 101325
    T_o = 273.15

    N_o = e_o / (k_b * T_o)  # Number density in m^-3 in the measurement conditions of the RI

    wv_o = np.array([288.24, 263.21, 252.93, 252.49, 252.00, 251.69, 251.51,
                     250.77, 243.59, 221.74, 221.16, 220.87, 212.48, 205.88,
                     198.90, 198.64, 198.32, 198.06, 197.92, 197.76, 191.85,
                     184.30])

    n_o = 1. + np.array([291.6, 298.6, 302.4, 302.7, 302.7, 302.9, 303.0, 303.3,
                         306.5, 320.1, 320.6, 320.8, 328.5, 336.1, 346.2, 346.7,
                         347.1, 347.6, 347.9, 348.2, 361.2, 379.2]) * 1E-6

    f_n = interp1d(wv_o, n_o, bounds_error=False, fill_value='extrapolate')

    alpha = (3. * eps_o) * (1. / N_o) * (np.power(f_n(x), 2) - 1.) / (np.power(f_n(x), 2) + 2.)

    return alpha

def zhang_O2(x):
    """
    
    zhang_O2:
        
        Original formula corresponds to T = 293.15K (20C), e = 1013.25 hPa
        
        Valid in the range 400-1800nm
        
        The correction of Kren is applied to the original formula of Zhang
        
        J. Zhang, Z. H. Lu, and L. J. Wang. Precision refractive index 
        measurements of air, N2, O2, Ar, and CO2 with a frequency comb, 
        Appl. Opt. 47, 3143-3151 (2008)
        
        P. Křen. Comment on "Precision refractive index measurements of air,
        N2, O2, Ar, and CO2 with a frequency comb", 
        Appl. Opt. 50, 6484-6485 (2011)

    """

    x = x * 1E-3  # the original formula is expressed in μm

    e_o = 101325
    T_o = 293.15

    N_o = e_o / (k_b * T_o)  # Number density in the measurement conditions of the RI

    c1 = 1.181494E-4
    c2 = 9.708931E-3
    c3 = 75.4
    n_o = 1. + c1 + c2 / (c3 - np.power(x, -2))

    alpha = (3. * eps_o) * (1. / N_o) * (np.power(n_o, 2) - 1.) / (np.power(n_o, 2) + 2.)

    return alpha

def bideau_mehu_larsen_Ar(x):
    """
    
    bideau_mehu_larsen_Ar:
        
        Original formula corresponds to T = 273.15K (0C), e = 1013.25 hPa
        
        Valid in the range 141-567nm
        
        A. Bideau-Mehu, Y. Guern, R. Abjean, A. Johannin-Gilles. 
        Measurement of refractive indices of neon, argon, krypton and xenon 
        in the 253.7-140.4 nm wavelength range. Dispersion relations and 
        estimated oscillator strengths of the resonance lines. J. Quant. 
        Spectrosc. Rad. Transfer 25, 395-402 (1981)
        
        T. Larsén. Beitrag zur Dispersion der Edelgase. Z. 
        Physik 88, 389-394 (1934)
        
        *Sellmeier formula is derived by the authors of ref. 1 
        using their own data in the 0.1404-0.2537 μm range combined with 
        data from ref. 2 at longer wavelengths.

    """

    x = x * 1E-3  # the original formula is expressed in μm

    e_o = 101325
    T_o = 273.15

    N_o = e_o / (k_b * T_o)  # Number density in the measurement conditions of the RI

    c1 = 2.50141E-3
    c2 = 91.012
    c3 = 5.00283E-4
    c4 = 87.892
    c5 = 5.22343E-2
    c6 = 214.02

    n_o = 1. + c1 / (c2 - np.power(x, -2)) + c3 / (c4 - np.power(x, -2)) + \
          c5 / (c6 - np.power(x, -2))

    alpha = (3. * eps_o) * (1. / N_o) * (np.power(n_o, 2) - 1.) / (np.power(n_o, 2) + 2.)

    return alpha


def boerzsoenyi_Ar(x):
    """
    
    boerzsoenyi_Ar:
        
        Original formula corresponds to T = 273.15K (0C), e = 1000.00 hPa (1000 mbar)
        
        Valid in the range 400-1000nm
                  
        A. Börzsönyi, Z. Heiner, M. P. Kalashnikov, A. P. Kovács, 
        and K. Osvay, Dispersion measurement of inert gases and gas 
        mixtures at 800 nm, Appl. Opt. 47, 4856-4863 (2008)

    """

    x = x * 1E-3  # the original formula is expressed in μm

    e_o = 100000
    T_o = 273.15

    N_o = e_o / (k_b * T_o)  # Number density in the measurement conditions of the RI

    c1 = 20332.29E-8
    c2 = 206.12E-6
    c3 = 34458.31E-8
    c4 = 8.066E-3

    n_o = np.sqrt(1. + c1 * np.power(x, 2) / (np.power(x, 2) - c2) + \
                  c3 * np.power(x, 2) / (np.power(x, 2) - c4))

    alpha = (3. * eps_o) * (1. / N_o) * (np.power(n_o, 2) - 1.) / (np.power(n_o, 2) + 2.)

    return alpha


def peck_and_fisher_Ar(x):
    """
    
    peck_fisher_Ar:
        
        Original formula corresponds to T = 273.15K (0C), e = 1013.25 hPa
        
        Valid in the range 468-2058nm
                  
        E. R. Peck and D. J. Fisher. Dispersion of argon, J. 
        Opt. Soc. Am. 54, 1362-1364 (1964)
        
    """

    x = x * 1E-3  # the original formula is expressed in μm

    e_o = 101325
    T_o = 273.15

    N_o = e_o / (k_b * T_o)  # Number density in the measurement conditions of the RI

    c1 = 6.7867E-5
    c2 = 3.0182943E-2
    c3 = 144

    n_o = 1. + c1 + c2 / (c3 - np.power(x, -2))

    alpha = (3. * eps_o) * (1. / N_o) * (np.power(n_o, 2) - 1.) / (np.power(n_o, 2) + 2.)

    return alpha


def bideau_mehu_larsen_CO2(x):
    """
    
    bideau_mehu_CO2:
        

    """

    x = x * 1E-3  # the original formula is expressed in μm

    e_o = 101325
    T_o = 273.15

    N_o = e_o / (k_b * T_o)  # Number density in the measurement conditions of the RI

    c1 = 6.99100E-2
    c2 = 166.175
    c3 = 1.44720E-3
    c4 = 79.609
    c5 = 6.42941E-5
    c6 = 56.3064
    c7 = 5.21306E-5
    c8 = 46.0196
    c9 = 1.46847E-6
    c10 = 0.0584738

    n_o = 1. + c1 / (c2 - np.power(x, -2)) + c3 / (c4 - np.power(x, -2)) + \
          c5 / (c6 - np.power(x, -2)) + c7 / (c8 - np.power(x, -2)) + \
          c9 / (c10 - np.power(x, -2))

    alpha = (3. * eps_o) * (1. / N_o) * (np.power(n_o, 2) - 1.) / (np.power(n_o, 2) + 2.)

    return alpha


def old_CO2(x):
    """
    
    old_CO2:
    
    """

    x = x * 1E-3  # the original formula is expressed in μm

    e_o = 101325
    T_o = 273.15

    N_o = e_o / (k_b * T_o)  # Number density in the measurement conditions of the RI

    c1 = 1.54489E-6
    c2 = 5.84738E-2
    c3 = 8.3091927E-2
    c4 = 210.9241
    c5 = 2.8764190E-3
    c6 = 60.122959

    n_o = 1. + c1 / (c2 - np.power(x, -2)) + c3 / (c4 - np.power(x, -2)) + \
          c5 / (c6 - np.power(x, -2))

    alpha = (3. * eps_o) * (1. / N_o) * (np.power(n_o, 2) - 1.) / (np.power(n_o, 2) + 2.)

    return alpha

def cidor_H2O(x):
    """
    cidor_H2O:
        
        Original formula corresponds to T = 293.15K (20C), e = 13.33 hPa
        
        Valid in the range 350-1200nm (claimed by Cidor to be able to work 
                                       out of the original measurement range)
        
        P. E. Ciddor, “Refractive index of air: new equations for the
        visible and near infrared,” Appl. Opt. 35, 1566 –1573 (1996).

    """

    x = x * 1E-3  # the original formula is expressed in μm

    e_o = 1333
    T_o = 293.15

    N_o = e_o / (k_b * T_o)  # Number density in the measurement conditions of the RI

    c1 = 1.022
    c2 = 295.235
    c3 = 2.6422
    c4 = 0.032380
    c5 = 0.004028

    n_o = 1. + c1 * (c2 + c3 * np.power(x, -2) - c4 * np.power(x, -4) + \
                     c5 * np.power(x, -6)) * 1e-8

    alpha = (3. * eps_o) * (1. / N_o) * (np.power(n_o, 2) - 1.) / (np.power(n_o, 2) + 2.)

    return alpha


functions = {"griesmann_and_burnett_N2" : griesmann_and_burnett_N2, 
             "boerzsoenyi_N2"           : boerzsoenyi_N2, 
             "peck_and_khanna_N2"       : peck_and_khanna_N2,
             "smith_O2"                 : smith_O2, 
             "zhang_O2"                 : zhang_O2,
             "bideau_mehu_larsen_Ar"    : bideau_mehu_larsen_Ar, 
             "boerzsoenyi_Ar"           : boerzsoenyi_Ar,
             "peck_and_fisher_Ar"       : peck_and_fisher_Ar,
             "bideau_mehu_larsen_CO2"   : bideau_mehu_larsen_CO2,
             "old_CO2"                  : old_CO2, 
             "cidor_H2O"                : cidor_H2O}

def alpha_gas(wavelength, gas):
    
    """ Provides the isotropic polarizability per gas. 
    
    Parameters
    ----------
    wavelength : float
       Wavelength in vacuum [nm]
    
    gas : string
        must be one of : "N2", "O2", "Ar", "CO2", "H2O"
        
    Returns
    -------
    alpha : float
       Isotropic polarizability of N2 in C2 m2 J-1 (SI)
       
    """

    if not isinstance(wavelength, int) and not isinstance(wavelength, float):
        raise Exception(("-- Error: The provided wavelength must be a scalar float or integer"))
    
    if wavelength < 350. or wavelength > 1200.:
        raise Exception(f'The selected wavelength ({wavelength} nm) is out of ARC limits: 350 to 1200 nm')
        
    if gas not in supported_gases:
        raise  Exception(f'The provided gas ({gas}) is not supported. Please use one of: {supported_gases}')
        
    for method in valid_methods[gas]:
        if wavelength >= lower_limit[method] and wavelength <= upper_limit[method]:
            alpha = functions[method](wavelength)

    return alpha  