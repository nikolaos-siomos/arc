""" Utility functions used in all modules. """

from .refractive_index import molar_fraction_water_vapour, compressibility_of_moist_air, \
                              saturation_vapor_pressure
from .constants import k_b
from .us_std import Atmosphere

import os

def number_density_at_pt(pressure, temperature, relative_humidity, ideal=False):
    """ Calculate the number density for a given temperature and pressure,
    taking into account the compressibility of air.
    
    Parameters
    ----------
    pressure: float or array
       Pressure in hPa
    temperature: float or array
       Temperature in K
    relative_humidity: float or array (?)
       ? The relative humidity of air (Check)
    ideal: boolean
       If False, the compressibility of air is considered. If True, the 
       compressibility is set to 1.
    
    Returns
    -------
    n: array or array
       Number density of the atmosphere [m-3]   
    """
    Xw = molar_fraction_water_vapour(pressure, temperature, relative_humidity)
        
    if ideal:
        Z = 1
    else:    
        Z = compressibility_of_moist_air(pressure, temperature, Xw)

    p_pa = pressure * 100.  # Pressure in pascal

    n = p_pa / (Z * temperature * k_b)
    
    return n
    
    
def rh_to_pressure(rh, temperature):
    """ Convert relative humidity to water vapour partial pressure.
    
    Parameters
    ----------
    rh: float
       Relative humidity from 0 to 100 [%]
    temperature: float
       Temperature [K]
       
    Returns
    -------
    p_wv: float
       Water vapour pressure [hPa].
    """
    svp = saturation_vapor_pressure(temperature)
    h = rh / 100.
    
    p_wv = h * svp
    return p_wv  # Previously / 100. This seems to be a bug (SVP already in hPA)/


def pressure_to_rh(partial_pressure, temperature):
    """ Convert water vapour partial pressure to relative humidity.

    Parameters
    ----------
    partial_pressure: float
       Water vapour partial pressure [hPa] 
    temperature: float
       Temperature [K]

    Returns
    -------
    rh: float
       Relative humidity from 0 to 100 [%].
    """
    svp = saturation_vapor_pressure(temperature)

    rh = partial_pressure / svp * 100

    return rh



def atmospheric_optical_depth_us_std(wavelength, zmin=0, zmax=50000):
    r"""
    Calculate the Rayleigh optical depth at certain altitude.

    Parameters
    ----------
    wavelength: float
        The wavelength of the radiation [nanometers].
    zmin, zmax: float
        The min and max altitude to take into account. [m]

    Returns
    -------
    tau: float
        The atmospheric optical depth.
    """
    z = np.linspace(zmin, zmax, 1000)

    atm = Atmosphere()
    temperatures = np.array([atm.temperature(z_i) for z_i in z])
    pressures = np.array([atm.pressure(z_i) for z_i in z])

    optical_depth = atmospheric_optical_depth(wavelength, pressures, temperatures, z / 1000.)  # / 1000, convert to Km, to be compatible with base OD function
    return optical_depth
