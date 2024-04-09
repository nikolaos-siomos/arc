import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

class BaseFilter:
    """ Base class containing only plotting methods. Actual filter functions should be implemented in the
    subclasses. """

    def plot(self, xmin, xmax):
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
        filter_wavelengths = np.linspace(xmin, xmax, 1000)

        filter_efficiency = self(filter_wavelengths)

        line_1, = ax.plot(filter_wavelengths, filter_efficiency, color_str,
                          label=label)
        ax.set_ylabel('Filter efficiency')
        ax.yaxis.label.set_color('green')
        ax.tick_params(axis='y', colors='green')
        ax.set_ylim(0, 1.1)
        return label, line_1, ax

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Filter functions should be implemented in the subclasses. ")


class GaussianFilter(BaseFilter):
    def __init__(self, wavelength, fwhm, shift = 0., transmittance=1, off_band_transmittance=0):
        '''
        This simple class represents a gausian filter function. To generate
        a new filter use::

           my_filter = FilterFunction(wavelegnth, fwhm)

        with

        wavelegnth - The central wavelength of the filter in nm
        fwhm       - The fwhm of the filter in nm
        shift: float
            Shifts the wavelength axis to higher wavelengths for positive shift values and to lower ones for
            negative shift values. The unit is [nm].

        If the the filter is called with a wavelegnth (in nm) as an argument
        it will return the  efficiency at this wavelength, for example::

           my_filter = FilterFunction(532, 5)
           my_filter(532) # Will return 1.0
           my_filter(535) # Will return 0.3685
        '''
        self.shift = shift 
        self.wavelength = wavelength + self.shift
        self.fwhm = fwhm
        self.c = fwhm / 2.354820045031

        self.tranmittance = transmittance
        self.off_band_transmittance = off_band_transmittance

    def __call__(self, wavelength):
        value = self.tranmittance * np.exp(-(wavelength - self.wavelength)
                                            ** 2 / (2 * self.c ** 2)) + self.off_band_transmittance
        return value

class LorentzianFilter(BaseFilter):
    def __init__(self, wavelength, fwhm, shift = 0., transmittance=1, off_band_transmittance=0):
        '''
        This simple class represents a gausian filter function. To generate
        a new filter use::

           my_filter = FilterFunction(wavelegnth, fwhm)

        with

        wavelegnth - The central wavelength of the filter in nm
        fwhm       - The fwhm of the filter in nm
        shift: float
            Shifts the wavelength axis to higher wavelengths for positive shift values and to lower ones for
            negative shift values. The unit is [nm].

        If the the filter is called with a wavelegnth (in nm) as an argument
        it will return the  efficiency at this wavelength, for example::

           my_filter = FilterFunction(532, 5)
           my_filter(532) # Will return 1.0
           my_filter(535) # Will return 0.3685
        '''
        self.shift = shift
        self.wavelength = wavelength + self.shift 
        self.fwhm = fwhm

        self.gamma = fwhm / 2.

        self.tranmittance = transmittance
        self.off_band_transmittance = off_band_transmittance

    def __call__(self, wavelength):
        
        # value_max = 1 / (2. * np.pi * self.fwhm)  
        value = (self.tranmittance) * self.gamma**2 / ((wavelength - self.wavelength)** 2 + self.gamma**2)
        
        return value

class DoubleLorentzianFilter(BaseFilter):
    def __init__(self, wavelength, fwhm, shift = 0., transmittance=1, off_band_transmittance=0):
        '''
        This simple class represents a gausian filter function. To generate
        a new filter use::

           my_filter = FilterFunction(wavelegnth, fwhm)

        with

        wavelegnth - The central wavelength of the filter in nm
        fwhm       - The fwhm of the filter in nm
        
        shift: float
            Shifts the wavelength axis to higher wavelengths for positive shift values and to lower ones for
            negative shift values. The unit is [nm].

        If the the filter is called with a wavelegnth (in nm) as an argument
        it will return the  efficiency at this wavelength, for example::

           my_filter = FilterFunction(532, 5)
           my_filter(532) # Will return 1.0
           my_filter(535) # Will return 0.3685
        '''
        self.shift = shift
        self.wavelength = wavelength + self.shift
        self.fwhm = fwhm

        self.gamma = fwhm / 2.

        self.tranmittance = transmittance
        self.off_band_transmittance = off_band_transmittance

    def __call__(self, wavelength):
        
        value_max = 1 / (2. * np.pi * self.fwhm)  
        value = (self.tranmittance) * self.gamma**2 / ((wavelength - self.wavelength)** 2 + self.gamma**2)
        
        return value


class SquareFilter(BaseFilter):
    def __init__(self, wavelength, width, shift = 0., transmittance=1, off_band_transmittance=0):
        '''
        This simple class represents a square filter function. To generate
        a new filter use::

           my_filter = FilterFunction(wavelegnth, width, transmittacnce)
           
         Parameters:
         
             shift: float
                 Shifts the wavelength axis to higher wavelengths for positive shift values and to lower ones for
                 negative shift values. The unit is [nm].

        '''
        self.shift = shift
        self.wavelength = wavelength + self.shift
        self.width = width
        self.min_wavelength = wavelength - width / 2.
        self.max_wavelength = wavelength + width / 2.
        self.transmittance = transmittance
        self.off_band_transmittance = off_band_transmittance

    def __call__(self, wavelength):
        w = np.array(wavelength)
        values = np.ones_like(w)
        transmitting_bins = (w > self.min_wavelength) & (w < self.max_wavelength)
        values[transmitting_bins] = self.transmittance
        values[~transmitting_bins] = self.off_band_transmittance
        return values


class CombinedFilter(BaseFilter):
    def __init__(self, filters):
        """
        A combination of several filters. The results will be the multiplication of all filters.

        Parameters
        ----------
        filters : list
           A list of filters
        """
        self.filters = filters

    def __call__(self, wavelength):
        filter_values = [f(wavelength) for f in self.filters]
        values = np.prod(filter_values, axis=0)
        return values


class FileFilter(BaseFilter):
    def __init__(self, file_path, shift = 0., interpolation='linear', off_band_transmittance = 0):
        """
        A filter with transmission given by a text file.

        Currently assumes a simple two-column, ';'-delimited file. The first columne should be the wavelength
        in nm and the second the transmittance at the specified wavelength (0 - 1).

        Parameters
        ----------
        file_path : str
           Path to filter function.
        shift: float
            Shifts the wavelength axis to higher wavelengths for positive shift values and to lower ones for
            negative shift values. The unit is [nm].
        interpolation : str
           The kind of interpolation between provided values. One of 'linear', 'nearest', 'zero', 'slinear',
           'quadratic', 'cubic'. Corresponds to 'kind' argument of interp1d.
         off_band_transmittance: int
             ...????

        """
        self.file_path = file_path
        self.shift = shift
        self.interpolation = interpolation
        self.off_band_transmittance = off_band_transmittance
        self.read_data()

    def read_data(self):
        # TODO: Make this more flexible using pandas?
        data = np.loadtxt(self.file_path, delimiter=';')
        self.wavelengths = data[:, 0]*1000 + self.shift
        self.transmittance = data[:, 1]
        self.transmission_function = interp1d(self.wavelengths, self.transmittance, kind=self.interpolation,bounds_error=False,fill_value=self.off_band_transmittance)

    def __call__(self, wavelength):
        return self.transmission_function(wavelength)


class CustomFilter(BaseFilter):
    def __init__(self, wavelengths, transmittances, interpolation='linear', off_band_transmittance = 0):
        """
        A filter with transmission given by a two arrays.

        This is a thin wrapper around numpy's interp1d.

        Parameters
        ----------
        wavelengths : numpy.array
           Wavelength [nm]
        transmittances : numpy.array
            Filter transmittance at the corresponding wavelength (from 0 to 1).
        interpolation : str
            The kind of interpolation between provided values. One of 'linear', 'nearest', 'zero', 'slinear',
            'quadratic', 'cubic'. Corresponds to 'kind' argument of interp1d.
        off_band_transmittance : float scalar
            Fill transmittance value for the regions out of the provided filter spectrum
        """
        
        self.wavelength = wavelengths
        self.transmittances = transmittances
        self.interpolation = interpolation
        self.off_band_transmittance = off_band_transmittance
        self.transimssion_function = interp1d(wavelengths, transmittances, kind=interpolation, bounds_error=False,fill_value=off_band_transmittance)

    def __call__(self, wavelength):
        return self.transimssion_function(wavelength)
