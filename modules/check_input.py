#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 18:18:52 2025

@author: nikos
"""
import os
import numpy as np

def check_molar_fractions(dictionary):
    
    dict_name = 'molar_fractions'
    keys = [key for key in dictionary.keys()]
    vals = [val for val in dictionary.values()]
    
    allowed_keys = ['N2', 'O2', 'Ar', 'CO2', 'H2O']
    allowed_values = [0., 1.]
    
    is_allowed = [key in allowed_keys for key in keys]
    print(is_allowed)
    if len(keys) == 0:
        raise Exception(f"--Error: The {dict_name} dictionary is empty. Please create a dictionary including all of the following keys:\n{allowed_keys}")
    elif all(is_allowed) and len(is_allowed) < len(allowed_keys):
        raise Exception(f"--Error: The {dict_name} dictionary has less keys than expected:\n{keys}\nPlease create a dictionary including all of the following keys:\n{allowed_keys}")
    elif all(is_allowed) == False:
        raise Exception(f"--Error: The {dict_name} dictionary contains non valid keys:\n{keys}\nPlease create a dictionary including all of the following keys:\n{allowed_keys}")
    else:
        is_invalid = [(val < allowed_values[0]) or (val > allowed_values[1]) \
                      for val in vals]
        if np.any(is_invalid):
            raise Exception(f"--Error: The {dict_name} dictionary contains non valid values:\n{dictionary}\nPlease create a dictionary using values within the following range:\n{allowed_values}")
        else:
            msg = f"--Error: The {dict_name} dictionary contains non valid values:\n{dictionary}\nThe sum of all values must be 1 with an accuracy better than 3 digits."
            np.testing.assert_almost_equal(np.sum(vals), 1., decimal = 3, err_msg = msg, verbose = False)
            
    return()

def check_filter_parameters(filter_parameters):
    
    default_parameters = {'transmission_shape' : None, 
                          'AOI' : 0., 
                          'ref_index_IF' : 2., 
                          'filter_path' : None,
                          'filter_file_delimiter' : ' ',
                          'filter_file_header_rows' : 0,
                          'wavelengths' : None,
                          'transmissions' : None,
                          'extra_shift' : 0., 
                          'central_wavelength' : None, 
                          'bandwidth' : None,
                          'peak_transmission' : 1.,
                          'off_band_transmission' : 0.}
    
    allowed_parameter_types = {'transmission_shape' : [str, type(None)], 
                               'AOI' : [float, np.float64, int, np.int64], 
                               'ref_index_IF' : [float, np.float64, int, np.int64], 
                               'filter_path' : [str, type(None)],
                               'filter_file_delimiter' : [str],
                               'filter_file_header_rows' : [int],
                               'wavelengths' : [np.ndarray, type(None)],
                               'transmissions' : [np.ndarray, type(None)],
                               'extra_shift' : [float, int], 
                               'central_wavelength' : [float, np.float64, int, np.int64, type(None)], 
                               'bandwidth' : [float, np.float64, int, np.int64, type(None)],
                               'peak_transmission' : [float, np.float64, int, np.int64],
                               'off_band_transmission' : [float, np.float64, int, np.int64]}

    check_parameter_type(parameters = filter_parameters,
                         allowed_parameter_types = allowed_parameter_types) 
        
    filter_parameters,default_used = \
        fill_missing_parameters(parameters = filter_parameters,
                                default_parameters = default_parameters)    
    
    check_parameter_values(filter_parameters = filter_parameters, 
                           default_used = default_used)

    return(filter_parameters)

def check_parameter_type(parameters, allowed_parameter_types):
    
    for key in parameters.keys():
        if type(parameters[key]) not in allowed_parameter_types[key]:
            raise Exception(f"--Error: The type of parameter {key} ({type(parameters[key])}) is wrong. Please use one of: {allowed_parameter_types[key]}")
    
    return

def check_parameter_values(filter_parameters, default_used):
    
    allowed_shapes = ['Gaussian', 'Lorentzian', 'Tophat', 'Custom']
    
    non_custom_parameters = ['central_wavelength', 'bandwidth', 'peak_transmission', 'off_band_transmission']
    
    custom_parameters = ['filter_path', 'filter_file_delimiter', 
                         'filter_file_header_rows', 'wavelengths', 
                         'transmissions', 'extra_shift']

    file_parameters = ['filter_file_delimiter', 'filter_file_header_rows']

    allowed_delimiters = [' ', ',', ':', ';', '\t']
    
    if filter_parameters['transmission_shape'] not in allowed_shapes:
        raise Exception(f"-- Error: The provided transmission_shape parameter ({filter_parameters['transmission_shape']}) was not recognized. Please use one of: {allowed_shapes}")
    
    if filter_parameters['ref_index_IF'] < 1. or filter_parameters['ref_index_IF'] > 3:
        raise Exception(f"-- Error: The provided ref_index_IF parameter {filter_parameters['ref_index_IF']} is not realistic")

    if filter_parameters['transmission_shape'] == 'Custom':
        
        if filter_parameters['filter_path'] is None and (filter_parameters['wavelengths'] is None or filter_parameters['transmissions'] is None):
            raise Exception("-- Error: The transmission_shape parameter is set to Custom. Providing either filter_path parameter or both the wavelengths and transmissions parameters is mandatory.")

        if filter_parameters['filter_path'] is not None and filter_parameters['wavelengths'] is not None and filter_parameters['transmissions'] is not None:
            raise Exception("-- Error: The transmission_shape parameter is set to Custom. The filter_path cannot be provided together with the wavelengths and transmissions parameters.")
                 
        if filter_parameters['filter_path'] is not None: 
            if not os.path.exists(filter_parameters['filter_path']):
                raise Exception(f"-- Error: The provided filter_path parameter does not correspond to an existing path:\n{filter_parameters['filter_path']}")

            if filter_parameters['filter_file_delimiter'] not in allowed_delimiters:
                raise Exception(f"-- Error: The provided filter_file_delimiter parameter ({filter_parameters['filter_file_delimiter']}) is not supported. Please use one of the allowed delimiters: {allowed_delimiters}")

            if filter_parameters['filter_file_header_rows'] < 0:
                raise Exception(f"-- Error: The provided filter_file_header_rows parameter ({filter_parameters['filter_file_header_rows']}) is negative.")

        else:
            
            for key in file_parameters:
                if filter_parameters[key] is not None and not default_used[key]:
                    print(f"-- Warning: The {key} parameter was provided but will be ignored because the filter_path is not provided ")

        if filter_parameters['wavelengths'] is not None:
            if (filter_parameters['wavelengths'] <= 0.).any():
                raise Exception("-- Error: Negative and/or zero values detected in the provided wavelengths parameter")
            
            if (filter_parameters['wavelengths'] < 200.).any() or (filter_parameters['wavelengths'] > 3000.).any() :
                raise Exception("-- Error: Values outside of the 200 - 3000 nm range detected in the provided wavelengths parameter")

        if filter_parameters['transmissions'] is not None:
            if (filter_parameters['transmissions'] <= 0.).any():
                raise Exception("-- Error: Negative and/or zero values detected in the provided transmissions parameter")

        if filter_parameters['transmissions'] is not None:
            if (filter_parameters['transmissions'] > 1.).any():
                raise Exception("-- Error: Values larger than 1. detected in the provided transmissions parameter")
                                
        for key in non_custom_parameters:
            if filter_parameters[key] is not None and not default_used[key]:
                print(f"-- Warning: The {key} parameter was provided but will be ignored because the transmission_shape is set to Custom ")

    else:
        
        if filter_parameters['central_wavelength'] is None:
            raise Exception("-- Error: The central_wavelength is a mandatory parameter when the transmission_shape is not set to 'Custom'")

        if filter_parameters['bandwidth'] is None:
            raise Exception("-- Error: The bandwidth is a mandatory parameter when the transmission_shape is not set to 'Custom'")

        if filter_parameters['peak_transmission'] is None:
            raise Exception("-- Error: The peak_transmission is a mandatory parameter when the transmission_shape is not set to 'Custom'")

        if filter_parameters['off_band_transmission'] is None and filter_parameters['transmission_shape'] == 'Tophat':
            raise Exception("-- Error: The off_band_transmission is a mandatory parameter when the transmission_shape is set to 'Tophat'")

        if filter_parameters['central_wavelength'] is not None:
            if filter_parameters['central_wavelength'] <= 0.:
                raise Exception(f"-- Error: The provided central_wavelength parameter ({filter_parameters['central_wavelength']}) is zero or negative")

            if filter_parameters['central_wavelength'] < 200. or filter_parameters['central_wavelength'] > 3000.:
                raise Exception(f"-- Error: The provided central_wavelength parameter ({filter_parameters['central_wavelength']}) is outside of the 200 - 3000 nm range")

        if filter_parameters['bandwidth'] is not None:
            if filter_parameters['bandwidth'] <= 0.:
                raise Exception(f"-- Error: The provided bandwidth {filter_parameters['bandwidth']} is zero or negative")

            if filter_parameters['bandwidth'] < 0.05 or filter_parameters['bandwidth'] > 100.:
                raise Exception(f"-- Error: The provided bandwidth parameter ({filter_parameters['bandwidth']}) is outside of the 0.05 - 100 nm range")

        if filter_parameters['peak_transmission'] is not None:
            if filter_parameters['peak_transmission'] <= 0.:
                raise Exception(f"-- Error: The provided peak_transmission parameter ({filter_parameters['peak_transmission']}) is zero or negative")
            
            if filter_parameters['peak_transmission'] > 1.:
                raise Exception(f"-- Error: The provided peak_transmission parameter ({filter_parameters['peak_transmission']}) is larger than 1.")

        if filter_parameters['off_band_transmission'] is not None:
            if filter_parameters['off_band_transmission'] < 0.:
                raise Exception(f"-- Error: The provided off_band_transmission parameter ({filter_parameters['off_band_transmission']}) is negative")
            
            if filter_parameters['off_band_transmission'] >= filter_parameters['peak_transmission']:
                raise Exception(f"-- Error: The provided off_band_transmission parameter ({filter_parameters['peak_transmission']}) is larger than or equal to the provided off_band_transmission parameter  ({filter_parameters['off_band_transmission']})")

        for key in custom_parameters:
            if filter_parameters[key] is not None and not default_used[key]:
                print(f"-- Warning: The {key} parameter was provided but will be ignored because the transmission_shape is not set to Custom ")
    
    return()

def fill_missing_parameters(parameters, default_parameters):
    
    default_used = dict()
    
    for key in default_parameters.keys():
        if key not in parameters.keys():
            parameters[key] = default_parameters[key]
            default_used[key] = True
        else:
            default_used[key] = False
            
    return(parameters, default_used)