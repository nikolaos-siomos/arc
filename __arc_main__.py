import numpy as np
from modules.make_gas import N2, O2, Ar, CO2, H2O
from modules.functions import raman_lines, xsection_polarized
from modules.filters import get_filter_transmission
from modules.check_input import check_molar_fractions, check_filter_parameters

class arc:
    def __init__(self, incident_wavelength, temperature = 288.15, 
                 molar_fractions = None, max_J = 100, 
                 backscattering = False, mode = 'rotational_raman',
                 filter_parameters = None):
        
        """
        This class calculates the volume depolarization ratio of the molecular
        backscatter signal detected with a polarization lidar.


        Input Parameters
        ----------
        incident_wavelength: float
            The wavelength of the radiation incedent to the molecules, in air (nm) - Mandatory
        
        temperature: float
            The atmospheric temperature (K). Defaults to 288.15 K (15°C)
        
        molar_fractions: dictionary with the gas names as indexes and 
            Molar fractions of atmospheric gases. If not provided the following
            molar fractions will be used by default:
        	{'N2' : 0.780796,  'O2' : 0.209448, 'Ar' : 0.009339, 'CO2': 0.000416, 'H2O': 0.}
        
        max_J: int
            Maximum rotational quantum number (number of lines considered per branch) 
        
        mode: string
            Choose among: rotational_raman and vibrational_raman
                rotational_raman: Corresponds to elastic and pure rotational Raman 
                                  lidar channel applications.
                vibrational_raman_N2: Corresponds to N2 vibrational Raman (V = 1)
                                      lidar channel applications.
                vibrational_raman_O2: Corresponds to O2 vibrational Raman (V = 1)
                                      lidar channel applications.
        
        backscattering: bool
            If set to True, backscattering cross-sections will be calculated 
            instead of scattering cross sections. Defaults to False

        filter_parameters: dictionary
            Dictionary entries that describe the interference filter (IF) 
            can be included under 	the following 	keys: 
        	
            transmission_shape: str 
                Use one of: 'Gaussian', 'Lorentzian', 'Tophat', 'Custom'. 
        	
            AOI: float
        		Angle of incidence (AOI) of the incident light with respect to 
                the optical axis of the IF. Defaults to 0.
        	
            ref_index_IF: float
        		Effective refractive index of the IF. Defaults to 2.
        	
            extra_shift: float
        		A wavelength extra_shift in nanometers that can be applied to a
                filter in addition to the AoI extra_shift. It will be ignored if
                the transmission_shape is not 'Custom'
        	
            filter_path: str
        		The path to an ascii file with the transmission function of the
                interference filter. The file must have 2 columns and no header. 
                The first corresponds to the wavelength scale. The second to the
                transmission for each wavelength. It will be ignored if the 
                transmission_shape is not 'Custom'
        	
            filter_file_delimiter: str 
        		The delimiter used to parse the filter file.  It will be ignored
                if the transmission_shape is not 'Custom' and the filter_path 
                is not provided. Defaults to ' '
        	
            filter_file_header_rows: int 
        		The number of header lines to skip when parsing the filter file.
                It will be ignored if the transmission_shape is not 'Custom' and
                the filter_path is not provided. Defaults to 0
        	
            wavelengths: 1D float array
        		An array of wavelength values that correspond to the transmission
                curve of a filter. Cannot be given as input at the same time with
                filter_path
        	
            transmissions: 1D float array
        		The array of transmission values per wavelength. Cannot be given
                as input at the same time with filter_path
        	
            central_wavelength: float
        		The central wavelength of the filter. It will be ignored if the 
                transmission_shape is set to 'Custom'
        	
            bandwidth: float
        		The filter bandwidth in nanometers. It will be ignored if the 
                transmission_shape is set to 'Custom'
    		
            peak_transmission: float 
        		The maximum transmission value. It will be ignored if the 
                transmission_shape is set to 'Custom'. Defaults to 1.

            off_band_transmission: float 
        		The transmission values at the off-band part of a Tophat filter. 
                It will be ignored if the transmission_shape is not set to 
                'Tophat'. Defaults to 0.
                        
        """

        if backscattering == False and filter_parameters is not None:
            print("-- Warning: Interference filter properties were provided but the backscattering argument is set to False. IFs can be only applied for the effective molecular backscatter cross section calculation. The backscattering argument will be switched to True")
            backscattering = True 
        
        allowed_modes = ['rotational_raman', 'vibrational_raman_N2', 'vibrational_raman_O2'] 
        if mode not in allowed_modes:
            raise Exception(f"-- Mode variable ({mode}) not understood. Please selecton of: {allowed_modes}")
        
        # Assume dry air mixture if the relative concentrations are not provided
        if mode == 'rotational_raman':
            if molar_fractions is None:
                self.molar_fractions = {'N2' : 0.780796,
                                        'O2' : 0.209448,
                                        'Ar' : 0.009339,
                                        'CO2': 0.000416,
                                        'H2O': 0.}
            else:
                check_molar_fractions(molar_fractions)
                self.molar_fractions = molar_fractions
                
        if mode == 'vibrational_raman_N2':
            self.molar_fractions = {'N2' : 1.,
                                    'O2' : 0.,
                                    'Ar' : 0.,
                                    'CO2': 0.,
                                    'H2O': 0.}       
            if molar_fractions != None:
                print("--Warning: Providing molar_fractions only makes sense in rotational_raman mode. The provided molar_fractions will be ignored")
        
        if mode == 'vibrational_raman_O2':
            self.molar_fractions = {'N2' : 0.,
                                    'O2' : 1.,
                                    'Ar' : 0.,
                                    'CO2': 0.,
                                    'H2O': 0.}     
            if molar_fractions != None:
                print("--Warning: Providing molar_fractions only makes sense in rotational_raman mode. The provided molar_fractions will be ignored")


        # Check filter_parameters and store into class object
        if filter_parameters is not None:
            filter_parameters = check_filter_parameters(filter_parameters)          
        
        self.filter_parameters = filter_parameters
        
        # Load gas parameters and save into the class
        gas_parameters = dict()
        gas_parameters['N2']  = N2(incident_wavelength)
        gas_parameters['O2']  = O2(incident_wavelength)
        gas_parameters['Ar']  = Ar(incident_wavelength)
        gas_parameters['CO2'] = CO2(incident_wavelength)
        gas_parameters['H2O'] = H2O(incident_wavelength)   
        self.gas_parameters = gas_parameters        
            
        # Filter transmission
        self.filter_transmission = \
            get_filter_transmission(filter_parameters = filter_parameters)
            
        # Save the rest of input into the class
        self.temperature = temperature
        self.max_J = max_J
        
        self.incident_wavelength = float(incident_wavelength)

        self.backscattering = backscattering         
        
        if mode == 'rotational_raman':
            linear_molecules = ['N2', 'O2', 'CO2']
            all_molecules = ['N2', 'O2', 'Ar', 'CO2', 'H2O']
        elif mode == 'vibrational_raman_N2':
            linear_molecules = ['N2']
            all_molecules = ['N2']
        elif mode == 'vibrational_raman_O2':
            linear_molecules = ['O2']
            all_molecules = ['O2']
        
        self.linear_molecules = linear_molecules
        self.all_molecules = all_molecules
        
        xsection_pol = dict()
        lamda_pol = dict()

        xsection_depol_line = dict()
        lamda_depol_line = dict()
        
        for gas in linear_molecules:          
            for branch in ['O', 'Q', 'S']:
                xsection_depol_line[f'{gas}_{branch}'], \
                    lamda_depol_line[f'{gas}_{branch}'] = \
                        raman_lines(incident_wavelength = self.incident_wavelength, 
                                    temperature = temperature,
                                    max_J = self.max_J, 
                                    mode = mode,
                                    branch = branch,
                                    molecular_parameters = self.gas_parameters[gas],
                                    backscattering = backscattering)

        self.xsection_depol_line = xsection_depol_line
        self.lamda_depol_line = lamda_depol_line
        
        for gas in self.all_molecules:
            xsection_pol[gas], lamda_pol[gas] = \
                xsection_polarized(self.incident_wavelength, 
                                   molecular_parameters = self.gas_parameters[gas], 
                                   temperature = temperature,
                                   mode = mode,
                                   backscattering = backscattering)
        
        self.xsection_pol = xsection_pol
        self.lamda_pol = lamda_pol
        
    def cross_section(self, cross_section_type = 'full', normalize = False):

        """ Caclulate the molecular backscattering or scattering cross section 
        by summing over the lines and gases
    
        Parameters
        ----------
        
        cross_section_type: string
            Choose among:  
                main_line: Cross section including only the polarized part and 
                the Q branch (e.g. Cabannes line, pure vibrational line)
                
                full: Cross section including all ro-vibrational lines 
                (e.g. full Rayleigh spectrum, full vibrational spectrum)
                
                polarized: Cross section including only the polarized part
                    
                depolarized: Cross section including only the depolarized 
                (Raman) part
                
                O: Cross section including only the anti-Stokes branch

                Q: Cross section including only the unshifted Q branch

                S: Cross section including only the Stokes branch
                
                wings: Cross section including only the Stokes and 
                anti-Stokes branches
                    
                If a filter was provided to the class its transmittion will
                be normally applied on the respective lines
                
        normalize: bool
            If set to True:
                the molecular cross section will be normalized 
                with the filter transmission at the emitted wavelength. 
                
                This is necessary for elastic channel application.
                The aerosol backscatter cross-section must be weightened with 
                the filter transmission similar to the molecular cross-section,
                otherwise the molecular cross-section contribution will be biased.
           
                Since this effect is neglected for most current lidar algorithms
                a simple alternative is to normalize the molecular cross-section 
                to compensate for it. Please note that normalizing will result to
                biases when simulating lidar signals.
                        
            If the no filter is provided or the cross_section_type is not set
            to 'full' or 'filter' then the cross section won't be normalized
        
        Returns
        -------
        
        cross_section:
           The backscattering or total scattering cross section. The returned
           value depends on the cross_section_type, normalize, and 
           scattering parameters
           
           Units are either [m2sr-1] or [m2].  

        """
        
        allowed_cross_section_types = ['main_line', 'full', 'polarized', 
                                       'depolarized', 'O', 'Q', 'S', 'wings']
        
        if cross_section_type not in allowed_cross_section_types:
            raise Exception(f'-- Error: Cross section type: {cross_section_type} not understood. Please select on of: {allowed_cross_section_types}')
                   
        if normalize and cross_section_type not in ['full', 'filter']:
            print(f'-- Warning: the normalize parameter is set to True but the cross_section_type is {cross_section_type}. Normalization will not be applied')                   
        
        filter_transmission = self.filter_transmission
        
        if cross_section_type in ['main_line', 'Q']:
            branches = ['Q']
        elif cross_section_type == 'O':
            branches = ['O']
        elif cross_section_type == 'S':
            branches = ['S']
        elif cross_section_type == 'wings':
            branches = ['O', 'S']
        else:
            branches = ['O', 'Q', 'S']
            
        xsection_depol = 0.
        
        if cross_section_type in ['main_line', 'full', 'depolarized', 'Q', 'O', 'S', 'wings']:
            for gas in self.linear_molecules:
                molar_fraction = self.molar_fractions[gas]
                
                for branch in branches:
                    xsection_depol_line = self.xsection_depol_line[f'{gas}_{branch}']
                    
                    lamda_depol_line = self.lamda_depol_line[f'{gas}_{branch}']
                    
                    if filter_transmission:
                        transmission = filter_transmission(lamda_depol_line)
                    else:
                        transmission = np.ones(lamda_depol_line.size)
                    
                    xsection_depol = xsection_depol + \
                        molar_fraction * np.nansum(transmission * xsection_depol_line)
                    
        xsection_pol = 0.
        
        if cross_section_type in ['main_line', 'full', 'polarized']:
            for gas in self.all_molecules:
                molar_fraction = self.molar_fractions[gas]
    
                xsection_pol_gas = self.xsection_pol[gas]
    
                lamda_pol_gas = self.lamda_pol[gas]
            
                if filter_transmission:
                    transmission = filter_transmission(lamda_pol_gas)
                else:
                    transmission = 1.
                
                xsection_pol = xsection_pol + \
                    molar_fraction * transmission * xsection_pol_gas                

        xsection = xsection_pol + xsection_depol
                
        if normalize and filter_transmission:
            xsection = xsection / filter_transmission(self.incident_wavelength)

        return xsection

    def mldr(self, mldr_type = 'full'):

        """ Caclulate the molecular linear depolarization ratio 
        by summing over the lines and gases
    
        Parameters
        ----------
        
        mldr_type: string
            Choose among:  
                main_line: MLDR including only the polarized part and 
                the Q branch (e.g. Cabannes line, pure vibrational line)
                
                full: MLDR including all ro-vibrational lines 
                (e.g. full Rayleigh spectrum, full vibrational spectrum)
                
                depolarized: Fixed MLDR of the depolarized lines (0.75)

                polarized: Fixed MLDR of the polarized line at (0.)
                    
                If a filter was provided to the class its transmittion will
                be normally applied on the respective lines
        
        Returns
        -------
        
        mldr:
           The molecular linear depolarization ratio calculated
           by summing over the lines and gases,
          
        """
        
        if self.backscattering:
            allowed_mldr_types = ['main_line', 'full', 'depolarized', 'polarized']
            
            if mldr_type not in allowed_mldr_types:
                raise Exception(f'-- Error: MLDR type: {mldr_type} not understood. Please select on of: {allowed_mldr_types}')
               
            if mldr_type == 'full':
                cross_section_type = 'depolarized'
            elif mldr_type == 'main_line':
                cross_section_type = 'Q'       
            
            
            depol_p = 4. / 7.
            
            depol_s = 3. / 7.
            
            if mldr_type in ['main_line', 'full']:
                xsection_pol = arc.cross_section(self, 
                                                 cross_section_type = 'polarized')
                
                xsection_depol = \
                    arc.cross_section(self, 
                                      cross_section_type = cross_section_type)                
            
                mldr = depol_s * xsection_depol / (xsection_pol + xsection_depol * depol_p)
            
            if mldr_type == 'depolarized':
                mldr = 3. /4.
                
            if mldr_type == 'polarized':
                mldr = 0.
        
        else:
            raise Exception("-- Error: The mldr function only supports backscattering cross section calculations. Please switch to backscatter calculation when calling the arc class by setting: backscattering = True")
            
        return(mldr)