"""
Unit test of the refractive_index functions.

The test is performed using the tabular values found in the referenced paper.
"""

import unittest
import numpy as np
from __arc_main__ import arc

# Get the data path
#current_path = os.path.dirname(__file__)
#data_base_path = os.path.join(current_path, '../data/bucholtz_tabular_values/')

wavelengths = np.array([355., 532., 1064.])
T1 = 293.15
T2 = 213.15

tolerance = 1E-12

class cross_sections(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def test_Cabannes_xsection_N2(self):

        true_values = np.array([3.26450E-31, 6.15242E-32, 3.73940E-33])
        
        calculated = np.nan * np.zeros(wavelengths.size)
        
        for i in range(len(wavelengths)):
            rrb = arc(wavelength = wavelengths[i], 
                      temperature = T1, 
                      relative_concentrations = [1., 0., 0., 0., 0.],
                      backscattering = True)
            
            calculated[i] = rrb.cross_section(cross_section_type = 'main_line')

        np.testing.assert_allclose(true_values, calculated, rtol=tolerance)
        
        pass

    # def test_Cabannes_xsection_O2(self):

    #     true_values = np.array([2.77881E-31, 5.13345E-32, 3.08499E-33])

    #     calculated = np.nan * np.zeros(wavelengths.size)
        
    #     for i in range(len(wavelengths)):
    #         rrb = arc(wavelength = wavelengths[i], 
    #                   temperature = T1, 
    #                   relative_concentrations = [0., 1., 0., 0., 0.],
    #backscattering = True)
            
    #         calculated[i] = rrb.cross_section(cross_section_type = 'main_line')

    #     np.testing.assert_allclose(true_values, calculated, rtol=tolerance)
    
    #     pass
    
    def test_rr_xsection_N2(self):

        true_values = np.array([6.16092E-33, 1.11582E-33, 6.60025E-35])

        calculated = np.nan * np.zeros(wavelengths.size)
        
        for i in range(len(wavelengths)):
            rrb = arc(wavelength = wavelengths[i], 
                      temperature = T1, 
                      relative_concentrations = [1., 0., 0., 0., 0.],
                      backscattering = True)
            
            calculated[i] = rrb.cross_section(cross_section_type = 'wings')

        np.testing.assert_allclose(true_values, calculated, rtol=tolerance)
      
        pass
    
    # def test_rr_xsection_O2(self):

    #     true_values = np.array([1.63587E-32, 2.67720E-33, 1.52284E-34])

    #     calculated = np.nan * np.zeros(wavelengths.size)
        
    #     for i in range(len(wavelengths)):
    #         rrb = arc(wavelength = wavelengths[i], 
    #                   temperature = T1, 
    #                   relative_concentrations = [0., 1., 0., 0., 0.],
    #backscattering = True)
            
    #         calculated[i] = rrb.cross_section(cross_section_type = 'depolarized')

    #     np.testing.assert_allclose(true_values, calculated, rtol=tolerance)
    
    #     pass
    
    def test_Stokes_J8_xsection_N2(self):

        true_values = np.array([3.74279E-34, 6.74960E-35, 3.94056E-36])

        calculated = np.nan * np.zeros(wavelengths.size)
        
        for i in range(len(wavelengths)):
            rrb = arc(wavelength = wavelengths[i], 
                      temperature = T1, 
                      relative_concentrations = [1., 0., 0., 0., 0.],
                      backscattering = True)
            
            calculated[i] = rrb.xsection_depol_line['N2_S'][8]

        np.testing.assert_allclose(true_values, calculated, rtol=tolerance)
    
        pass
    
    def test_antiStokes_J8_xsection_N2(self):

        true_values = np.array([3.00708E-34, 5.47506E-35, 3.28990E-36])

        calculated = np.nan * np.zeros(wavelengths.size)
        
        for i in range(len(wavelengths)):
            rrb = arc(wavelength = wavelengths[i], 
                      temperature = T1, 
                      relative_concentrations = [1., 0., 0., 0., 0.],
                      backscattering = True)
            
            calculated[i] = rrb.xsection_depol_line['N2_O'][8]

        np.testing.assert_allclose(true_values, calculated, rtol=tolerance)

        pass

    def test_Stokes_J18_xsection_N2(self):

        true_values = np.array([5.46857E-035, 9.80604E-036, 5.62742E-037])

        calculated = np.nan * np.zeros(wavelengths.size)
        
        for i in range(len(wavelengths)):
            rrb = arc(wavelength = wavelengths[i], 
                      temperature = T1, 
                      relative_concentrations = [1., 0., 0., 0., 0.],
                      backscattering = True)
            
            calculated[i] = rrb.xsection_depol_line['N2_S'][18]

        np.testing.assert_allclose(true_values, calculated, rtol=tolerance)
    
        pass
    
    def test_antiStokes_J18_xsection_N2(self):

        true_values = np.array([5.11607E-035, 9.36696E-036, 5.72283E-37])

        calculated = np.nan * np.zeros(wavelengths.size)
        
        for i in range(len(wavelengths)):
            rrb = arc(wavelength = wavelengths[i], 
                      temperature = T1, 
                      relative_concentrations = [1., 0., 0., 0., 0.],
                      backscattering = True)
            
            calculated[i] = rrb.xsection_depol_line['N2_O'][18]

        np.testing.assert_allclose(true_values, calculated, rtol=tolerance)

        pass

    # def test_Stokes_J9_xsection_O2(self):

    #     true_values = np.array([1.28548E-33, 2.09640E-34, 1.17982E-35])

    #     calculated = np.nan * np.zeros(wavelengths.size)
        
    #     for i in range(len(wavelengths)):
    #         rrb = arc(wavelength = wavelengths[i], 
    #                   temperature = T1, 
    #                   relative_concentrations = [0., 1., 0., 0., 0.],
   # backscattering = True)
            
    #         calculated[i] = rrb.xsection_depol_line['O2_S'][9]

    #     np.testing.assert_allclose(true_values, calculated, rtol=tolerance)
        
    #     pass
    
    # def test_antiStokes_J9_xsection_O2(self):

    #     true_values = np.array([1.05563E-33, 1.73492E-34, 9.99358E-36])

    #     calculated = np.nan * np.zeros(wavelengths.size)
        
    #     for i in range(len(wavelengths)):
    #         rrb = arc(wavelength = wavelengths[i], 
    #                   temperature = T1, 
    #                   relative_concentrations = [0., 1., 0., 0., 0.],
    #backscattering = True)
    #
            
    #         calculated[i] = rrb.xsection_depol_line['O2_O'][9]

    #     np.testing.assert_allclose(true_values, calculated, rtol=tolerance)
        
    #     pass

    # def test_Stokes_J8_lamda_N2(self):

    #     true_values = np.array([355.95485, 534.14727, 1072.62389])

    #     calculated = np.nan * np.zeros(wavelengths.size)
        
    #     for i in range(len(wavelengths)):
    #         rrb = arc(wavelength = wavelengths[i], 
    #                   temperature = T1, 
    #                   relative_concentrations = [0.79, 0., 0., 0., 0.],
    #backscattering = True)
            
    #         calculated[i] = rrb.lamda_depol_line['N2_S'][8]

    #     np.testing.assert_allclose(true_values, calculated, rtol=tolerance)
      
    #     pass
    
    # def test_antiStokes_J8_lamda_N2(self):

    #     true_values = np.array([354.24963, 530.31661, 1057.28769])

    #     calculated = np.nan * np.zeros(wavelengths.size)
        
    #     for i in range(len(wavelengths)):
    #         rrb = arc(wavelength = wavelengths[i], 
    #                   temperature = T1, 
    #                   relative_concentrations = [0.79, 0., 0., 0., 0.])
            
    #         calculated[i] = rrb.lamda_depol_line['N2_O'][8]

    #     np.testing.assert_allclose(true_values, calculated, rtol=tolerance)
      
    #     pass

class mldr(unittest.TestCase):
    
    def setUp(self):
        pass       
    def test_mldr_Cabannes(self):

        true_values = np.array([0.00380840, 0.00352867, 0.00339854])

        calculated = np.nan * np.zeros(wavelengths.size)
        
        for i in range(len(wavelengths)):
            rrb = arc(wavelength = wavelengths[i], 
                      temperature = T1, 
                      relative_concentrations = [0.80, 0.20, 0., 0., 0.],
                      backscattering = True)
            
            calculated[i] = rrb.mldr(mldr_type = 'main_line')

        np.testing.assert_allclose(true_values, calculated, rtol=tolerance)
      
        pass
    
    def test_mldr_Rayleigh(self):

        true_values = np.array([0.0150050, 0.0139182, 0.0134118])

        calculated = np.nan * np.zeros(wavelengths.size)
        
        for i in range(len(wavelengths)):
            rrb = arc(wavelength = wavelengths[i], 
                      temperature = T1, 
                      relative_concentrations = [0.80, 0.20, 0., 0., 0.],
                      backscattering = True)
            
            calculated[i] = rrb.mldr(mldr_type = 'full')

        np.testing.assert_allclose(true_values, calculated, rtol=tolerance)
      
        pass
if __name__ == "__main__":
    unittest.main()
    