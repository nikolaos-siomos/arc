import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import os

class BaseFilter:
    """ Base class containing only plotting methods. Actual filter functions should be implemented in the
    subclasses. """

    def plot(self, xmin = None, xmax = None):
        
        fig = plt.figure()
        ax = plt.subplot(111)
        self.draw_plot(ax, xmin, xmax, twin_axis=False)
        plt.draw()
        plt.show()
    
    def draw_plot(self, main_axis, xmin, xmax, twin_axis=True, color_str='-g', label='Filter'):
        if twin_axis:
            ax = main_axis.twinx()
        else:
            ax = main_axis
            
        if xmin == None or xmax == None:
            filter_wavelengths = \
                np.linspace(self.central_wavelength - 4. * self.bandwidth, 
                            self.central_wavelength + 4. * self.bandwidth, 
                            10000)
        else:
            filter_wavelengths = np.linspace(xmin, xmax, 10000)
    
        filter_transmission = self(filter_wavelengths)
    
        line_1, = ax.plot(filter_wavelengths, filter_transmission, color_str,
                          label=label)
        ax.set_ylabel('Interference Filter Transmission')
        ax.set_ylim(0, 1.1 * np.nanmax(filter_transmission))
        return label, line_1, ax
    
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Filter functions should be implemented in the subclasses. ")


class GaussianFilter(BaseFilter):
    def __init__(self, central_wavelength, bandwidth, extra_shift = 0., AOI = 0., ref_index_IF = 2, peak_transmission = 1.):
        '''
        This class creates an interference filter transmission curve with a 
        Gaussian shape.
        
        Input Parameters
        ----------
        central_wavelength: float
           The central wavelength of the filter [nm]
                
        bandwidth: float
           The bandwith of the filter (full width at half maximum) [nm]
           
        AOI: float
            Angle of incident light with respect to the filters optical axis. Deaults to: 0. [rad]
            
        ref_index_IF: float
            Effective refractive index of the filter. Defaults to: 2. 
            
        peak_transmission: float
            The transmission at the filter's peak. Defaults to: 1.

        Calling
        ----------
        If the filter is called with the wavelegnth [nm] as an argument
        it will return the  transmission at this wavelength, for example::
        
           my_filter = GaussianFilter(532, 5)
           my_filter(532) # Will return 1.0
        '''
        
        AOI_shift = \
            AOI_wavelength_shift(wavelength = central_wavelength,
                                 AOI = AOI, 
                                 ref_index_IF = ref_index_IF)

        self.central_wavelength = central_wavelength
        self.shifted_central_wavelength = central_wavelength + AOI_shift
        self.bandwidth = bandwidth
        self.c = bandwidth / 2.354820045031
        self.peak_transmission = peak_transmission
        
    def __call__(self, wavelength):
        values = self.peak_transmission * \
                np.exp(-(wavelength - self.shifted_central_wavelength) ** 2 / (2 * self.c ** 2))
        return values

class LorentzianFilter(BaseFilter):
    def __init__(self, central_wavelength, bandwidth, AOI = 0., ref_index_IF = 2, peak_transmission = 1.):
        '''
        This class creates an interference filter transmission curve with a 
        Lorentzian shape
        
        Input Parameters
        ----------
        central_wavelength: float
           The central wavelength of the filter [nm]
                
        bandwidth: float
           The bandwith of the filter (full width at half maximum) [nm]
        
        AOI: float
            Angle of incident light with respect to the filters optical axis. Deaults to: 0. [rad]
            
        ref_index_IF: float
            Effective refractive index of the filter. Defaults to: 2. 
            
        peak_transmission: float
            The transmission at the filter's peak. Defaults to: 1.

        Calling
        ----------
        If the filter is called with the wavelegnth [nm] as an argument
        it will return the  transmission at this wavelength, for example::
        
           my_filter = LorentzianFilter(532, 5)
           my_filter(532) # Will return 1.0
        '''

        AOI_shift = \
            AOI_wavelength_shift(wavelength = central_wavelength,
                                 AOI = AOI, 
                                 ref_index_IF = ref_index_IF)

        self.central_wavelength = central_wavelength
        self.shifted_central_wavelength = central_wavelength + AOI_shift
        self.bandwidth = bandwidth
        self.peak_transmission = peak_transmission        
        self.gamma = bandwidth / 2.

    def __call__(self, wavelength):
        values = (self.peak_transmission) * \
            self.gamma**2 / ((wavelength - self.shifted_central_wavelength)** 2 + self.gamma**2)
        return values


class TophatFilter(BaseFilter):
    def __init__(self, central_wavelength, bandwidth, AOI = 0., ref_index_IF = 2, peak_transmission = 1., off_band_transmission = 0.):
        
        '''
        This class creates an interference filter transmission curve with a 
        Lorentzian shape
        
        Input Parameters
        ----------
        central_wavelength: float
           The central wavelength of the filter [nm]
                
        bandwidth: float
           The bandwith of the filter [nm]
        
        AOI: float
            Angle of incident light with respect to the filters optical axis. Deaults to: 0. [rad]
            
        ref_index_IF: float
            Effective refractive index of the filter. Defaults to: 2. 

        peak_transmission: float
            The maximum constant transmission of the filter. Defaults to: 1.
        
        off_band_transmission: float 
    		The transmission values at the off-band part of a Tophat filter. 
            It will be ignored if the transmission_shape is not set to 
            'Tophat'. Defaults to 0.

        Calling
        ----------
        If the filter is called with the wavelegnth [nm] as an argument
        it will return the transmission at this wavelength, for example::
        
           my_filter = TophatFilter(532, 5)
           my_filter(532) # Will return 1.0
        '''

        AOI_shift = \
            AOI_wavelength_shift(wavelength = central_wavelength,
                                 AOI = AOI, 
                                 ref_index_IF = ref_index_IF)

        shifted_central_wavelength = central_wavelength + AOI_shift
        
        self.central_wavelength = central_wavelength
        self.shifted_central_wavelength = shifted_central_wavelength
        
        self.bandwidth = bandwidth
        self.peak_transmission = peak_transmission
        self.off_band_transmission = off_band_transmission
        self.min_wavelength = shifted_central_wavelength - bandwidth / 2.
        self.max_wavelength = shifted_central_wavelength + bandwidth / 2.
        
    def __call__(self, wavelength):
        
        if type(wavelength) in [int, float]:
            if wavelength > self.min_wavelength and wavelength < self.max_wavelength:
                values = self.peak_transmission
            else:
                values = self.off_band_transmission
        else:
            w = np.array(wavelength)

            values = self.peak_transmission * np.ones(w.size)
    
            non_transmitting_bins = (w < self.min_wavelength) | (w > self.max_wavelength)
            
            if non_transmitting_bins.any():
                values[non_transmitting_bins] = self.off_band_transmission

        return values
    

class RealFilterFile(BaseFilter):
    def __init__(self, filter_path, filter_file_delimiter = ' ',
                 filter_file_header_rows = 0, AOI = 0., ref_index_IF = 2., 
                 extra_shift = 0., interpolation = 'linear', 
                 off_band_transmission = 0.):
        """
        This class creates an interference filter transmission curve based on 
        an external ascii file or by providing arrays of the transmission and
        the correpsonding wavelengths

        Parameters
        ----------
        filter_path: string
           The path to the input ascii file that contains the interference filter 
           transmission values per wavelengths
        
        filter_file_delimiter: (str) 
            The delimiter used to parse the filter file.
            It will be ignored if the transmission_shape is not 'Custom'
            and the filter_path is not provided
            Defaults to ' '
        
        filter_file_header_rows: (int) 
            The number of header lines to skip when parsing the filter 
            file.
            It will be ignored if the transmission_shape is not 'Custom'
            and the filter_path is not provided
            Defaults to 0
                    
        extra_shift: float
            Shift of the center wavelength of the filter. Positive for extra_shift to higher wavelengths
            negative for shifts to lower wavelengths. Defaults to: 0.
        
        AOI: float
            Angle of incident light with respect to the filters optical axis. Deaults to: 0. [rad]
            
        ref_index_IF: float
            Effective refractive index of the filter. Defaults to: 2. 
            
        interpolation : str
            The kind of interpolation between provided values. One of 'linear', 'nearest', 'zero', 'slinear',
            'quadratic', 'cubic'. Corresponds to 'kind' argument of interp1d.
       
        off_band_transmission : float scalar
            Fill transmission value for the regions out of the provided filter spectrum

        """
                 
        try:
            data = np.loadtxt(filter_path, delimiter = filter_file_delimiter, 
                              skiprows = filter_file_header_rows, dtype = float)
        except:
            raise Exception('--Error: The filter file could not be parsed. Please check the filter_file_delimiter and the filter_file_header_rows parameters and also the filter file format ')
        
        wavelengths = data[:,0] + extra_shift
        
        transmissions = data[:,1]

        if np.max(transmissions) > 10.:
            transmissions = transmissions / 100. 
            print('-- Warning: The transmission values provided in the IF ASCII file seem to be per cent and have been converted to normal fractions. Please make sure that this is correct')
    
        central_wavelength = np.average(wavelengths, weights = transmissions)
        bandwidth = np.max(wavelengths[transmissions >= 0.5]) - np.min(wavelengths[transmissions >= 0.5])
        
        AOI_shift = \
            AOI_wavelength_shift(wavelength = central_wavelength,
                                 AOI = AOI, 
                                 ref_index_IF = ref_index_IF)
            
        shifted_wavelengths = wavelengths + AOI_shift + extra_shift

        self.central_wavelength = central_wavelength
        self.shifted_central_wavelength = central_wavelength + AOI_shift + extra_shift
        self.bandwidth = bandwidth
                
        self.transmission_function = interp1d(shifted_wavelengths, 
                                              transmissions, 
                                              kind = interpolation, 
                                              bounds_error = False,
                                              fill_value = off_band_transmission)

    def __call__(self, wavelength):
        return self.transmission_function(wavelength)

class RealFilterArray(BaseFilter):
    def __init__(self, wavelengths, transmissions, AOI = 0., ref_index_IF = 2., 
                 extra_shift = 0., interpolation = 'linear', 
                 off_band_transmission = 0.):
        """
        This class creates an interference filter transmission curve based on 
        an external ascii file or by providing arrays of the transmission and
        the correpsonding wavelengths

        Parameters
        ----------
        wavelengths: (1D float array)
            An array of wavelength values that correspond to the 
            transmission curve of a filter
            Cannot be given as input at the same time with filter_path
        
        transmissions: (1D float array)
            The array of transmission values per wavelength
            Cannot be given as input at the same time with filter_path
       
        extra_shift: float
            Shift of the center wavelength of the filter. Positive for extra_shift to higher wavelengths
            negative for shifts to lower wavelengths. Defaults to: 0.
        
        AOI: float
            Angle of incident light with respect to the filters optical axis. Deaults to: 0. [rad]
            
        ref_index_IF: float
            Effective refractive index of the filter. Defaults to: 2. 
            
        interpolation: str
            The kind of interpolation between provided values. One of 'linear', 'nearest', 'zero', 'slinear',
            'quadratic', 'cubic'. Corresponds to 'kind' argument of interp1d.
       
        off_band_transmission: float scalar
            Fill transmission value for the regions out of the provided filter spectrum

        """
                        
        central_wavelength = np.average(wavelengths, weights = transmissions)
        bandwidth = np.max(wavelengths[transmissions >= 0.5]) - np.min(wavelengths[transmissions >= 0.5])
        
        AOI_shift = \
            AOI_wavelength_shift(wavelength = central_wavelength,
                                 AOI = AOI, 
                                 ref_index_IF = ref_index_IF)
            
        shifted_wavelengths = wavelengths + AOI_shift + extra_shift

        self.central_wavelength = central_wavelength
        self.shifted_central_wavelength = central_wavelength + AOI_shift + extra_shift
        self.bandwidth = bandwidth
            
        shifted_wavelengths = wavelengths + AOI_shift + extra_shift
        self.transmission_function = interp1d(shifted_wavelengths, 
                                              transmissions, 
                                              kind = interpolation, 
                                              bounds_error = False,
                                              fill_value = off_band_transmission)

    def __call__(self, wavelength):
        return self.transmission_function(wavelength)

def get_filter_transmission(filter_parameters = None):
    
    """
    Returns a function that provides the filter transmission when the wavelength 
    is given as input (scalar or array).
    
    Input Parameters
    ----------
    
    filter_parameters: dictionary
        Provide the interference filter 
        parameters listed with the following keys:
            
            transmission_shape (str) 
                Use one of: 'Gaussian', 'Lorentzian', 'Tophat', 'Custom'. 
            
            AOI (float)
                Angle of incidence (AOI) of the incident light with respect to 
                the optical axis of the IF.
                Defaults to 0.

            ref_index_IF (float)
                Effective refractive index of the IF.
                Defaults to 2.
            
            extra_shift (float)
                A wavelength extra_shift in nanometers that can be applied to a 
                filter in addition to the AoI extra_shift
                It will be ignored if the transmission_shape is not 'Custom'
            
            filter_path (str) 
                The path to an ascii file with the transmission function of the 
                interference filter. The file must have 2 columns and no header. 
                The first corresponds to the wavelength grid.
                The second to the transmission for each wavelength.
                It will be ignored if the transmission_shape is not 'Custom'
                
            filter_file_delimiter: (str) 
                The delimiter used to parse the filter file.
                It will be ignored if the transmission_shape is not 'Custom'
                and the filter_path is not provided
                Defaults to ' '
            
            filter_file_header_rows: (int) 
                The number of header lines to skip when parsing the filter 
                file.
                It will be ignored if the transmission_shape is not 'Custom'
                and the filter_path is not provided
                Defaults to 0
            
            wavelengths (1D float array)
                An array of wavelength values that correspond to the 
                transmission curve of a filter
                Cannot be given as input at the same time with filter_path
            
            transmissions (1D float array)
                The array of transmission values per wavelength
                Cannot be given as input at the same time with filter_path
                
            central_wavelength (float)
                The central wavelength of the filter. 
                It will be ignored if the transmission_shape is 'Custom'
            
            bandwidth (float)
                The filter bandwidth in nanometers.
                It will be ignored if the transmission_shape is 'Custom'

            peak_transmission (float)
               The maximum transmission value. 
               It will be ignored if the transmission_shape is 'Custom'
               Defaults to 1.
    
    """
    
    filter_transmission = None
                     
    if filter_parameters is not None:
        
        AOI = float(filter_parameters['AOI'])
        
        ref_index_IF = float(filter_parameters['ref_index_IF'])
        
        transmission_shape = filter_parameters['transmission_shape']
        
        if transmission_shape == 'Custom':
    
            filter_path = filter_parameters['filter_path']
        
            extra_shift = float(filter_parameters['extra_shift'])
            
            wavelengths = filter_parameters['wavelengths']
    
            transmissions = filter_parameters['transmissions']
                
            if filter_parameters['filter_path'] is not None:
                
                filter_file_delimiter = filter_parameters['filter_file_delimiter']
                
                filter_file_header_rows = filter_parameters['filter_file_header_rows']
            
                filter_transmission = \
                    RealFilterFile(filter_path = filter_path,
                                   filter_file_delimiter = filter_file_delimiter,
                                   filter_file_header_rows = filter_file_header_rows,
                                   AOI = AOI,
                                   ref_index_IF = ref_index_IF,
                                   extra_shift = extra_shift)
            else:
                filter_transmission = \
                    RealFilterArray(wavelengths = wavelengths,
                                    transmissions = transmissions,
                                    AOI = AOI,
                                    ref_index_IF = ref_index_IF,
                                    extra_shift = extra_shift)
        else:
                
            transmission_function = {'Gaussian' : GaussianFilter,
                                     'Lorentzian' : LorentzianFilter,
                                     'Tophat' : TophatFilter}

            central_wavelength = float(filter_parameters['central_wavelength'])
    
            bandwidth = float(filter_parameters['bandwidth'])
    
            peak_transmission = float(filter_parameters['peak_transmission'])

            off_band_transmission = float(filter_parameters['off_band_transmission'])

            if transmission_shape == 'Tophat':
                filter_transmission = \
                    transmission_function[transmission_shape](central_wavelength = central_wavelength, 
                                                              bandwidth = bandwidth,
                                                              peak_transmission = peak_transmission,
                                                              off_band_transmission = off_band_transmission,
                                                              AOI = AOI,
                                                              ref_index_IF = ref_index_IF)
            
            else:
                filter_transmission = \
                    transmission_function[transmission_shape](central_wavelength = central_wavelength, 
                                                              bandwidth = bandwidth,
                                                              peak_transmission = peak_transmission,
                                                              AOI = AOI,
                                                              ref_index_IF = ref_index_IF)
            
    return(filter_transmission)

def AOI_wavelength_shift(wavelength, AOI, ref_index_IF):
    """Returns the wavelength extra_shift to the new filter central wavelength 
        due to the AOI of light onto the IF.

    Parameters
    ----------
    wavelength : float
       Wavelenth of the incident light.

    AOI: float
        Angle of incidence (AOI) of the incident light with respect to 
        the optical axis of the IF.

    ref_index_IF : float
        Effective refractive index of the IF.
    
    Returns
    ----------
    aoi_wavelengh_shift : float
       Wavelenth extra_shift of the IF transmission curve in nm 
       (AOI = 0 corresponds to 0. extra_shift).    

    """
    
    ref_index_air = 1.
    
    aoi_wavelengh_shift = wavelength * \
        (np.sqrt(1. - ((ref_index_air / ref_index_IF) * np.sin(np.deg2rad(AOI)))**2) - 1.)
    
    return aoi_wavelengh_shift

