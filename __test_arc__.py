"""
Unit test of the refractive_index functions.

The test is performed using the tabular values found in the referenced paper.
"""

import unittest
import numpy as np
from __arc_main__ import arc
from modules.isotropic_polarizability import alpha_gas
from modules.anisotropic_polarizability import kings_factor
from modules.functions import placzek_teller

# Get the data path
#current_path = os.path.dirname(__file__)
#data_base_path = os.path.join(current_path, '../data/bucholtz_tabular_values/')

wavelengths = np.array([355., 532., 1064.])

T1 = 285.15

tolerance = 1E-5

c_dry = {'N2' : 0.780796, 
         'O2' : 0.209448, 
         'Ar' : 0.009339, 
         'CO2': 0.000416, 
         'H2O': 0.}

c_moist = {'N2' : 0.773065, 
           'O2' : 0.207374, 
           'Ar' : 0.009247, 
           'CO2' : 0.000412, 
           'H2O' : 0.009902}

LJ = 8
HJ = 19

class cross_sections_summed(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def test_scs_dry(self):

        label = 'Scattering Cross-Section (Dry Air)'
        
        true_values_ray = np.array([2.759695e-30, 5.168334e-31, 3.128966e-32])
        true_values_cab = np.array([2.660805e-30, 4.996334e-31, 3.028776e-32])
        true_values_wng = np.array([9.889007e-32, 1.719997e-32, 1.001891e-33])
        true_values_pol = np.array([2.626480e-30, 4.936611e-31, 2.993900e-32])
        true_values_dpl = np.array([1.332146e-31, 2.317228e-32, 1.350660e-33])
        true_values_O = np.array([4.164929e-32, 7.278993e-33, 4.309010e-34])
        true_values_Q = np.array([3.432451e-32, 5.972306e-33, 3.487693e-34])
        true_values_S = np.array([5.724078e-32, 9.920977e-33, 5.709895e-34])
        
        calculated_ray = np.nan * np.zeros(wavelengths.size)
        calculated_cab = np.nan * np.zeros(wavelengths.size)
        calculated_wng = np.nan * np.zeros(wavelengths.size)
        calculated_pol = np.nan * np.zeros(wavelengths.size)
        calculated_dpl = np.nan * np.zeros(wavelengths.size)
        calculated_O = np.nan * np.zeros(wavelengths.size)
        calculated_Q = np.nan * np.zeros(wavelengths.size)
        calculated_S = np.nan * np.zeros(wavelengths.size)
        
        for i in range(len(wavelengths)):
            rrb = arc(incident_wavelength = wavelengths[i], 
                      temperature = T1, 
                      molar_fractions = c_dry,
                      backscattering = False)
            
            calculated_ray[i] = rrb.cross_section(cross_section_type = 'full')
            calculated_cab[i] = rrb.cross_section(cross_section_type = 'main_line')
            calculated_wng[i] = rrb.cross_section(cross_section_type = 'wings')
            calculated_pol[i] = rrb.cross_section(cross_section_type = 'polarized')
            calculated_dpl[i] = rrb.cross_section(cross_section_type = 'depolarized')
            calculated_O[i] = rrb.cross_section(cross_section_type = 'O')
            calculated_Q[i] = rrb.cross_section(cross_section_type = 'Q')
            calculated_S[i] = rrb.cross_section(cross_section_type = 'S')

        np.testing.assert_allclose(true_values_ray, calculated_ray, rtol=tolerance, err_msg = 'Rayleigh ' + label)
        np.testing.assert_allclose(true_values_cab, calculated_cab, rtol=tolerance, err_msg = 'Cabannes ' + label)
        np.testing.assert_allclose(true_values_wng, calculated_wng, rtol=tolerance, err_msg = 'RR Wings ' + label)
        np.testing.assert_allclose(true_values_pol, calculated_pol, rtol=tolerance, err_msg = 'Polarized ' + label)
        np.testing.assert_allclose(true_values_dpl, calculated_dpl, rtol=tolerance, err_msg = 'Depolarized ' + label)
        np.testing.assert_allclose(true_values_O, calculated_O, rtol=tolerance, err_msg = 'AntiStokes ' + label)
        np.testing.assert_allclose(true_values_Q, calculated_Q, rtol=tolerance, err_msg = 'Unshifted ' + label)
        np.testing.assert_allclose(true_values_S, calculated_S, rtol=tolerance, err_msg = 'Stokes ' + label)
        
        pass

    def test_scs_moist(self):

        label = 'Scattering Cross-Section (Moist Air)'

        true_values_ray = np.array([2.752220e-30, 5.153627e-31, 3.119770e-32])
        true_values_cab = np.array([2.654309e-30, 4.983330e-31, 3.020573e-32])
        true_values_wng = np.array([9.791097e-32, 1.702968e-32, 9.919710e-34])
        true_values_pol = np.array([2.620324e-30, 4.924199e-31, 2.986041e-32])
        true_values_dpl = np.array([1.318956e-31, 2.294285e-32, 1.337287e-33])
        true_values_O = np.array([4.123693e-32, 7.206926e-33, 4.266347e-34])
        true_values_Q = np.array([3.398467e-32, 5.913175e-33, 3.453162e-34])
        true_values_S = np.array([5.667404e-32, 9.822750e-33, 5.653363e-34])
        
        calculated_ray = np.nan * np.zeros(wavelengths.size)
        calculated_cab = np.nan * np.zeros(wavelengths.size)
        calculated_wng = np.nan * np.zeros(wavelengths.size)
        calculated_pol = np.nan * np.zeros(wavelengths.size)
        calculated_dpl = np.nan * np.zeros(wavelengths.size)
        calculated_O = np.nan * np.zeros(wavelengths.size)
        calculated_Q = np.nan * np.zeros(wavelengths.size)
        calculated_S = np.nan * np.zeros(wavelengths.size)
        
        for i in range(len(wavelengths)):
            rrb = arc(incident_wavelength = wavelengths[i], 
                      temperature = T1, 
                      molar_fractions = c_moist,
                      backscattering = False)
            
            calculated_ray[i] = rrb.cross_section(cross_section_type = 'full')
            calculated_cab[i] = rrb.cross_section(cross_section_type = 'main_line')
            calculated_wng[i] = rrb.cross_section(cross_section_type = 'wings')
            calculated_pol[i] = rrb.cross_section(cross_section_type = 'polarized')
            calculated_dpl[i] = rrb.cross_section(cross_section_type = 'depolarized')
            calculated_O[i] = rrb.cross_section(cross_section_type = 'O')
            calculated_Q[i] = rrb.cross_section(cross_section_type = 'Q')
            calculated_S[i] = rrb.cross_section(cross_section_type = 'S')

        np.testing.assert_allclose(true_values_ray, calculated_ray, rtol=tolerance, err_msg = 'Rayleigh ' + label)
        np.testing.assert_allclose(true_values_cab, calculated_cab, rtol=tolerance, err_msg = 'Cabannes ' + label)
        np.testing.assert_allclose(true_values_wng, calculated_wng, rtol=tolerance, err_msg = 'RR Wings ' + label)
        np.testing.assert_allclose(true_values_pol, calculated_pol, rtol=tolerance, err_msg = 'Polarized ' + label)
        np.testing.assert_allclose(true_values_dpl, calculated_dpl, rtol=tolerance, err_msg = 'Depolarized ' + label)
        np.testing.assert_allclose(true_values_O, calculated_O, rtol=tolerance, err_msg = 'AntiStokes ' + label)
        np.testing.assert_allclose(true_values_Q, calculated_Q, rtol=tolerance, err_msg = 'Unshifted ' + label)
        np.testing.assert_allclose(true_values_S, calculated_S, rtol=tolerance, err_msg = 'Stokes ' + label)
        
        pass

    def test_bcs_dry(self):

        label = 'Backscattering Cross-Section (Dry Air)'

        true_values_ray = np.array([3.246439e-31, 6.086264e-32, 3.686561e-33])
        true_values_cab = np.array([3.163810e-31, 5.942548e-32, 3.602846e-33])
        true_values_wng = np.array([8.262893e-33, 1.437167e-33, 8.371431e-35])
        true_values_pol = np.array([3.135130e-31, 5.892645e-32, 3.573704e-33])
        true_values_dpl = np.array([1.113092e-32, 1.936191e-33, 1.128562e-34])
        true_values_O = np.array([3.480063e-33, 6.082061e-34, 3.600451e-35])
        true_values_Q = np.array([2.868031e-33, 4.990241e-34, 2.914189e-35])
        true_values_S = np.array([4.782830e-33, 8.289606e-34, 4.770980e-35])
        
        calculated_ray = np.nan * np.zeros(wavelengths.size)
        calculated_cab = np.nan * np.zeros(wavelengths.size)
        calculated_wng = np.nan * np.zeros(wavelengths.size)
        calculated_pol = np.nan * np.zeros(wavelengths.size)
        calculated_dpl = np.nan * np.zeros(wavelengths.size)
        calculated_O = np.nan * np.zeros(wavelengths.size)
        calculated_Q = np.nan * np.zeros(wavelengths.size)
        calculated_S = np.nan * np.zeros(wavelengths.size)
        
        for i in range(len(wavelengths)):
            rrb = arc(incident_wavelength = wavelengths[i], 
                      temperature = T1, 
                      molar_fractions = c_dry,
                      backscattering = True)
            
            calculated_ray[i] = rrb.cross_section(cross_section_type = 'full')
            calculated_cab[i] = rrb.cross_section(cross_section_type = 'main_line')
            calculated_wng[i] = rrb.cross_section(cross_section_type = 'wings')
            calculated_pol[i] = rrb.cross_section(cross_section_type = 'polarized')
            calculated_dpl[i] = rrb.cross_section(cross_section_type = 'depolarized')
            calculated_O[i] = rrb.cross_section(cross_section_type = 'O')
            calculated_Q[i] = rrb.cross_section(cross_section_type = 'Q')
            calculated_S[i] = rrb.cross_section(cross_section_type = 'S')

        np.testing.assert_allclose(true_values_ray, calculated_ray, rtol=tolerance, err_msg = 'Rayleigh ' + label)
        np.testing.assert_allclose(true_values_cab, calculated_cab, rtol=tolerance, err_msg = 'Cabannes ' + label)
        np.testing.assert_allclose(true_values_wng, calculated_wng, rtol=tolerance, err_msg = 'RR Wings ' + label)
        np.testing.assert_allclose(true_values_pol, calculated_pol, rtol=tolerance, err_msg = 'Polarized ' + label)
        np.testing.assert_allclose(true_values_dpl, calculated_dpl, rtol=tolerance, err_msg = 'Depolarized ' + label)
        np.testing.assert_allclose(true_values_O, calculated_O, rtol=tolerance, err_msg = 'AntiStokes ' + label)
        np.testing.assert_allclose(true_values_Q, calculated_Q, rtol=tolerance, err_msg = 'Unshofted ' + label)
        np.testing.assert_allclose(true_values_S, calculated_S, rtol=tolerance, err_msg = 'Stokes ' + label)
                
        pass

    def test_bcs_moist(self):

        label = 'Backscattering Cross-Section (Moist Air)'

        true_values_ray = np.array([3.237989e-31, 6.069531e-32, 3.676063e-33])
        true_values_cab = np.array([3.156178e-31, 5.927238e-32, 3.593177e-33])
        true_values_wng = np.array([8.181083e-33, 1.422937e-33, 8.288547e-35])
        true_values_pol = np.array([3.127782e-31, 5.877829e-32, 3.564324e-33])
        true_values_dpl = np.array([1.102072e-32, 1.917021e-33, 1.117388e-34])
        true_values_O = np.array([3.445607e-33, 6.021844e-34, 3.564804e-35])
        true_values_Q = np.array([2.839635e-33, 4.940833e-34, 2.885336e-35])
        true_values_S = np.array([4.735476e-33, 8.207531e-34, 4.723743e-35])
        
        calculated_ray = np.nan * np.zeros(wavelengths.size)
        calculated_cab = np.nan * np.zeros(wavelengths.size)
        calculated_wng = np.nan * np.zeros(wavelengths.size)
        calculated_pol = np.nan * np.zeros(wavelengths.size)
        calculated_dpl = np.nan * np.zeros(wavelengths.size)
        calculated_O = np.nan * np.zeros(wavelengths.size)
        calculated_Q = np.nan * np.zeros(wavelengths.size)
        calculated_S = np.nan * np.zeros(wavelengths.size)
        
        for i in range(len(wavelengths)):
            rrb = arc(incident_wavelength = wavelengths[i], 
                      temperature = T1, 
                      molar_fractions = c_moist,
                      backscattering = True)
            
            calculated_ray[i] = rrb.cross_section(cross_section_type = 'full')
            calculated_cab[i] = rrb.cross_section(cross_section_type = 'main_line')
            calculated_wng[i] = rrb.cross_section(cross_section_type = 'wings')
            calculated_pol[i] = rrb.cross_section(cross_section_type = 'polarized')
            calculated_dpl[i] = rrb.cross_section(cross_section_type = 'depolarized')
            calculated_O[i] = rrb.cross_section(cross_section_type = 'O')
            calculated_Q[i] = rrb.cross_section(cross_section_type = 'Q')
            calculated_S[i] = rrb.cross_section(cross_section_type = 'S')

        np.testing.assert_allclose(true_values_ray, calculated_ray, rtol=tolerance, err_msg = 'Rayleigh ' + label)
        np.testing.assert_allclose(true_values_cab, calculated_cab, rtol=tolerance, err_msg = 'Cabannes ' + label)
        np.testing.assert_allclose(true_values_wng, calculated_wng, rtol=tolerance, err_msg = 'RR Wings ' + label)
        np.testing.assert_allclose(true_values_pol, calculated_pol, rtol=tolerance, err_msg = 'Polarized ' + label)
        np.testing.assert_allclose(true_values_dpl, calculated_dpl, rtol=tolerance, err_msg = 'Depolarized ' + label)
        np.testing.assert_allclose(true_values_O, calculated_O, rtol=tolerance, err_msg = 'AntiStokes ' + label)
        np.testing.assert_allclose(true_values_Q, calculated_Q, rtol=tolerance, err_msg = 'Unshifted ' + label)
        np.testing.assert_allclose(true_values_S, calculated_S, rtol=tolerance, err_msg = 'Stokes ' + label)
        
        pass

class cross_sections_lines(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def test_N2(self):

        label_xs = 'Backscattering Cross-Section'
        label_wv = 'Wavelength'

        true_values_xs_O_LJ = np.array([3.033099e-34, 5.524403e-35, 3.317783e-36])
        true_values_xs_O_HJ = np.array([1.733381e-35, 3.176520e-36, 1.942890e-37])
        true_values_xs_S_LJ = np.array([3.775172e-34, 6.810427e-35, 3.973955e-36])
        true_values_xs_S_HJ = np.array([1.838363e-35, 3.295791e-36, 1.887102e-37])
        
        true_values_wv_O_LJ = np.array([ 354.249631,  530.316613, 1057.287691])
        true_values_wv_O_HJ = np.array([ 353.157851,  527.873621, 1047.621520])
        true_values_wv_S_LJ = np.array([ 355.954855,  534.147270, 1072.623886])
        true_values_wv_S_HJ = np.array([ 357.062938,  536.646361, 1082.749194])

        calculated_xs_O_LJ = np.nan * np.zeros(wavelengths.size)
        calculated_xs_O_HJ = np.nan * np.zeros(wavelengths.size)
        calculated_xs_S_LJ = np.nan * np.zeros(wavelengths.size)
        calculated_xs_S_HJ = np.nan * np.zeros(wavelengths.size)
        
        calculated_wv_O_LJ = np.nan * np.zeros(wavelengths.size)
        calculated_wv_O_HJ = np.nan * np.zeros(wavelengths.size)
        calculated_wv_S_LJ = np.nan * np.zeros(wavelengths.size)
        calculated_wv_S_HJ = np.nan * np.zeros(wavelengths.size)
        
        for i in range(len(wavelengths)):
            rrb = arc(incident_wavelength = wavelengths[i], 
                      temperature = T1, 
                      backscattering = True)
            
            calculated_xs_O_LJ[i] = rrb.xsection_depol_line['N2_O'][LJ]
            calculated_xs_O_HJ[i] = rrb.xsection_depol_line['N2_O'][HJ]
            calculated_xs_S_LJ[i] = rrb.xsection_depol_line['N2_S'][LJ]
            calculated_xs_S_HJ[i] = rrb.xsection_depol_line['N2_S'][HJ]
            
            calculated_wv_O_LJ[i] = rrb.lamda_depol_line['N2_O'][LJ]
            calculated_wv_O_HJ[i] = rrb.lamda_depol_line['N2_O'][HJ]
            calculated_wv_S_LJ[i] = rrb.lamda_depol_line['N2_S'][LJ]
            calculated_wv_S_HJ[i] = rrb.lamda_depol_line['N2_S'][HJ]

        np.testing.assert_allclose(true_values_xs_O_LJ, calculated_xs_O_LJ, rtol=tolerance, err_msg = f'AntiStokes J{LJ} ' + label_xs)
        np.testing.assert_allclose(true_values_xs_O_HJ, calculated_xs_O_HJ, rtol=tolerance, err_msg = f'AntiStokes J{HJ} ' + label_xs)
        np.testing.assert_allclose(true_values_xs_S_LJ, calculated_xs_S_LJ, rtol=tolerance, err_msg = f'Stokes J{LJ} ' + label_xs)
        np.testing.assert_allclose(true_values_xs_S_HJ, calculated_xs_S_HJ, rtol=tolerance, err_msg = f'Stokes J{HJ} ' + label_xs)
        np.testing.assert_allclose(true_values_wv_O_LJ, calculated_wv_O_LJ, rtol=tolerance, err_msg = f'AntiStokes J{LJ} ' + label_wv)
        np.testing.assert_allclose(true_values_wv_O_HJ, calculated_wv_O_HJ, rtol=tolerance, err_msg = f'AntiStokes J{HJ} ' + label_wv)
        np.testing.assert_allclose(true_values_wv_S_LJ, calculated_wv_S_LJ, rtol=tolerance, err_msg = f'Stokes J{LJ} ' + label_wv)
        np.testing.assert_allclose(true_values_wv_S_HJ, calculated_wv_S_HJ, rtol=tolerance, err_msg = f'Stokes J{HJ} ' + label_wv)

        pass
    
    def test_O2(self):

        label_xs = 'Backscattering Cross-Section'
        label_wv = 'Wavelength'
        
        true_values_xs_O_LJ = np.array([0., 0., 0.])
        true_values_xs_O_HJ = np.array([2.871214e-34, 4.737866e-35, 2.762192e-36])
        true_values_xs_S_LJ = np.array([0., 0., 0.])
        true_values_xs_S_HJ = np.array([3.082538e-34, 5.006609e-35, 2.783047e-36])
        
        true_values_wv_O_LJ = np.array([ 354.457489,  530.782568, 1059.14139 ])
        true_values_wv_O_HJ = np.array([ 353.667365,  529.012791, 1052.117885])
        true_values_wv_S_LJ = np.array([ 355.689411,  533.549764, 1070.217169])
        true_values_wv_S_HJ = np.array([ 356.487694,  535.348024, 1077.47691 ])

        calculated_xs_O_LJ = np.nan * np.zeros(wavelengths.size)
        calculated_xs_O_HJ = np.nan * np.zeros(wavelengths.size)
        calculated_xs_S_LJ = np.nan * np.zeros(wavelengths.size)
        calculated_xs_S_HJ = np.nan * np.zeros(wavelengths.size)
        
        calculated_wv_O_LJ = np.nan * np.zeros(wavelengths.size)
        calculated_wv_O_HJ = np.nan * np.zeros(wavelengths.size)
        calculated_wv_S_LJ = np.nan * np.zeros(wavelengths.size)
        calculated_wv_S_HJ = np.nan * np.zeros(wavelengths.size)
        
        for i in range(len(wavelengths)):
            rrb = arc(incident_wavelength = wavelengths[i], 
                      temperature = T1, 
                      backscattering = True)
            
            calculated_xs_O_LJ[i] = rrb.xsection_depol_line['O2_O'][LJ]
            calculated_xs_O_HJ[i] = rrb.xsection_depol_line['O2_O'][HJ]
            calculated_xs_S_LJ[i] = rrb.xsection_depol_line['O2_S'][LJ]
            calculated_xs_S_HJ[i] = rrb.xsection_depol_line['O2_S'][HJ]
            
            calculated_wv_O_LJ[i] = rrb.lamda_depol_line['O2_O'][LJ]
            calculated_wv_O_HJ[i] = rrb.lamda_depol_line['O2_O'][HJ]
            calculated_wv_S_LJ[i] = rrb.lamda_depol_line['O2_S'][LJ]
            calculated_wv_S_HJ[i] = rrb.lamda_depol_line['O2_S'][HJ]

        np.testing.assert_allclose(true_values_xs_O_LJ, calculated_xs_O_LJ, rtol=tolerance, err_msg = f'AntiStokes J{LJ} ' + label_xs)
        np.testing.assert_allclose(true_values_xs_O_HJ, calculated_xs_O_HJ, rtol=tolerance, err_msg = f'AntiStokes J{HJ} ' + label_xs)
        np.testing.assert_allclose(true_values_xs_S_LJ, calculated_xs_S_LJ, rtol=tolerance, err_msg = f'Stokes J{LJ} ' + label_xs)
        np.testing.assert_allclose(true_values_xs_S_HJ, calculated_xs_S_HJ, rtol=tolerance, err_msg = f'Stokes J{HJ} ' + label_xs)
        np.testing.assert_allclose(true_values_wv_O_LJ, calculated_wv_O_LJ, rtol=tolerance, err_msg = f'AntiStokes J{LJ} ' + label_wv)
        np.testing.assert_allclose(true_values_wv_O_HJ, calculated_wv_O_HJ, rtol=tolerance, err_msg = f'AntiStokes J{HJ} ' + label_wv)
        np.testing.assert_allclose(true_values_wv_S_LJ, calculated_wv_S_LJ, rtol=tolerance, err_msg = f'Stokes J{LJ} ' + label_wv)
        np.testing.assert_allclose(true_values_wv_S_HJ, calculated_wv_S_HJ, rtol=tolerance, err_msg = f'Stokes J{HJ} ' + label_wv)

        pass
    
    def test_CO2(self):
        
        label_xs = 'Backscattering Cross-Section'
        label_wv = 'Wavelength'

        true_values_xs_O_LJ = np.array([1.493821e-33, 2.795009e-34, 1.683899e-35])
        true_values_xs_O_HJ = np.array([0., 0., 0.])
        true_values_xs_S_LJ = np.array([1.888227e-33, 3.526328e-34, 2.112527e-35])
        true_values_xs_S_HJ = np.array([0., 0., 0.])
        true_values_wv_O_LJ = np.array([ 354.852515,  531.668851, 1062.676229])
        true_values_wv_O_HJ = np.array([ 354.636494,  531.184063, 1060.741250])
        true_values_wv_S_LJ = np.array([ 355.186986,  532.420038, 1065.681481])
        true_values_wv_S_HJ = np.array([ 355.403655,  532.907033, 1067.634328])

        calculated_xs_O_LJ = np.nan * np.zeros(wavelengths.size)
        calculated_xs_O_HJ = np.nan * np.zeros(wavelengths.size)
        calculated_xs_S_LJ = np.nan * np.zeros(wavelengths.size)
        calculated_xs_S_HJ = np.nan * np.zeros(wavelengths.size)
        calculated_wv_O_LJ = np.nan * np.zeros(wavelengths.size)
        calculated_wv_O_HJ = np.nan * np.zeros(wavelengths.size)
        calculated_wv_S_LJ = np.nan * np.zeros(wavelengths.size)
        calculated_wv_S_HJ = np.nan * np.zeros(wavelengths.size)
        
        for i in range(len(wavelengths)):
            rrb = arc(incident_wavelength = wavelengths[i], 
                      temperature = T1, 
                      backscattering = True)
            
            calculated_xs_O_LJ[i] = rrb.xsection_depol_line['CO2_O'][LJ]
            calculated_xs_O_HJ[i] = rrb.xsection_depol_line['CO2_O'][HJ]
            calculated_xs_S_LJ[i] = rrb.xsection_depol_line['CO2_S'][LJ]
            calculated_xs_S_HJ[i] = rrb.xsection_depol_line['CO2_S'][HJ]
            
            calculated_wv_O_LJ[i] = rrb.lamda_depol_line['CO2_O'][LJ]
            calculated_wv_O_HJ[i] = rrb.lamda_depol_line['CO2_O'][HJ]
            calculated_wv_S_LJ[i] = rrb.lamda_depol_line['CO2_S'][LJ]
            calculated_wv_S_HJ[i] = rrb.lamda_depol_line['CO2_S'][HJ]

        np.testing.assert_allclose(true_values_xs_O_LJ, calculated_xs_O_LJ, rtol=tolerance, err_msg = f'AntiStokes J{LJ} ' + label_xs)
        np.testing.assert_allclose(true_values_xs_O_HJ, calculated_xs_O_HJ, rtol=tolerance, err_msg = f'AntiStokes J{HJ} ' + label_xs)
        np.testing.assert_allclose(true_values_xs_S_LJ, calculated_xs_S_LJ, rtol=tolerance, err_msg = f'Stokes J{LJ} ' + label_xs)
        np.testing.assert_allclose(true_values_xs_S_HJ, calculated_xs_S_HJ, rtol=tolerance, err_msg = f'Stokes J{HJ} ' + label_xs)
        np.testing.assert_allclose(true_values_wv_O_LJ, calculated_wv_O_LJ, rtol=tolerance, err_msg = f'AntiStokes J{LJ} ' + label_wv)
        np.testing.assert_allclose(true_values_wv_O_HJ, calculated_wv_O_HJ, rtol=tolerance, err_msg = f'AntiStokes J{HJ} ' + label_wv)
        np.testing.assert_allclose(true_values_wv_S_LJ, calculated_wv_S_LJ, rtol=tolerance, err_msg = f'Stokes J{LJ} ' + label_wv)
        np.testing.assert_allclose(true_values_wv_S_HJ, calculated_wv_S_HJ, rtol=tolerance, err_msg = f'Stokes J{HJ} ' + label_wv)

        pass

class mldr(unittest.TestCase):
    
    def setUp(self):
        pass      
    
    def test_mldr_dry(self):

        label = 'MLDR (Dry Air)'

        true_values_ray = np.array([0.014913, 0.013822, 0.013294])
        true_values_cab = np.array([0.00390 , 0.003612, 0.003479])
        true_values_pol = np.array([0., 0., 0.])
        true_values_dpl = np.array([0.75, 0.75, 0.75])
        
        calculated_ray = np.nan * np.zeros(wavelengths.size)
        calculated_cab = np.nan * np.zeros(wavelengths.size)
        calculated_pol = np.nan * np.zeros(wavelengths.size)
        calculated_dpl = np.nan * np.zeros(wavelengths.size)
        
        for i in range(len(wavelengths)):
            rrb = arc(incident_wavelength = wavelengths[i], 
                      temperature = T1, 
                      molar_fractions = c_dry, 
                      backscattering = True)
            
            calculated_ray[i] = rrb.mldr(mldr_type = 'full')
            calculated_cab[i] = rrb.mldr(mldr_type = 'main_line')
            calculated_pol[i] = rrb.mldr(mldr_type = 'polarized')
            calculated_dpl[i] = rrb.mldr(mldr_type = 'depolarized')

        np.testing.assert_allclose(true_values_ray, calculated_ray, atol=tolerance, err_msg = 'Rayleigh ' + label)
        np.testing.assert_allclose(true_values_cab, calculated_cab, atol=tolerance, err_msg = 'Cabannes ' + label)
        np.testing.assert_allclose(true_values_pol, calculated_pol, atol=tolerance, err_msg = 'Polarized ' + label)
        np.testing.assert_allclose(true_values_dpl, calculated_dpl, atol=tolerance, err_msg = 'Depolarized ' + label)
        
        pass

    def test_mldr_moist(self):

        label = 'MLDR (Moist Air)'

        true_values_ray = np.array([0.014803, 0.013722, 0.013199])
        true_values_cab = np.array([0.003871, 0.003585, 0.003453])
        true_values_pol = np.array([0., 0., 0.])
        true_values_dpl = np.array([0.75, 0.75, 0.75])
        
        calculated_ray = np.nan * np.zeros(wavelengths.size)
        calculated_cab = np.nan * np.zeros(wavelengths.size)
        calculated_pol = np.nan * np.zeros(wavelengths.size)
        calculated_dpl = np.nan * np.zeros(wavelengths.size)
        
        for i in range(len(wavelengths)):
            rrb = arc(incident_wavelength = wavelengths[i], 
                      temperature = T1, 
                      molar_fractions = c_moist, 
                      backscattering = True)
            
            calculated_ray[i] = rrb.mldr(mldr_type = 'full')
            calculated_cab[i] = rrb.mldr(mldr_type = 'main_line')
            calculated_pol[i] = rrb.mldr(mldr_type = 'polarized')
            calculated_dpl[i] = rrb.mldr(mldr_type = 'depolarized')

        np.testing.assert_allclose(true_values_ray, calculated_ray, atol=tolerance, err_msg = 'Rayleigh ' + label)
        np.testing.assert_allclose(true_values_cab, calculated_cab, atol=tolerance, err_msg = 'Cabannes ' + label)
        np.testing.assert_allclose(true_values_pol, calculated_pol, atol=tolerance, err_msg = 'Polarized ' + label)
        np.testing.assert_allclose(true_values_dpl, calculated_dpl, atol=tolerance, err_msg = 'Depolarized ' + label)
        
        pass
    
class polarizabilities(unittest.TestCase):

    def setUp(self):
        pass   

    def test_alpha_N2(self):

        true_values = np.array([2.023680e-40, 1.973573e-40, 1.945842e-40])

        calculated = np.nan * np.zeros(wavelengths.size)
        
        for i in range(len(wavelengths)):
            alpha_val = alpha_gas(wavelength = wavelengths[i], gas = "N2")
            
            calculated[i] = alpha_val

        np.testing.assert_allclose(true_values, calculated, rtol=tolerance)
      
        pass
    
    def test_kings_factor_N2(self):

        true_values = np.array([1.03651, 1.03512, 1.03428])

        calculated = np.nan * np.zeros(wavelengths.size)
        
        for i in range(len(wavelengths)):
            Fk_val = kings_factor(wavelength = wavelengths[i], gas = "N2")
            
            calculated[i] = Fk_val

        np.testing.assert_allclose(true_values, calculated, rtol=tolerance)
      
        pass

    def test_alpha_O2(self):

        true_values = np.array([1.85360E-40, 1.79126E-40, 1.75728E-40])

        calculated = np.nan * np.zeros(wavelengths.size)
        
        for i in range(len(wavelengths)):
            alpha_val = alpha_gas(wavelength = wavelengths[i], gas = "O2")
            
            calculated[i] = alpha_val

        np.testing.assert_allclose(true_values, calculated, rtol=tolerance)
      
        pass
    
    def test_kings_factor_O2(self):

        true_values = np.array([1.11611, 1.10270, 1.09734])

        calculated = np.nan * np.zeros(wavelengths.size)
        
        for i in range(len(wavelengths)):
            Fk_val = kings_factor(wavelength = wavelengths[i], gas = "O2")
            
            calculated[i] = Fk_val

        np.testing.assert_allclose(true_values, calculated, rtol=tolerance)
      
        pass

    def test_alpha_CO2(self):

        true_values = np.array([3.061248e-40, 2.972539e-40, 2.914831e-40])

        calculated = np.nan * np.zeros(wavelengths.size)
        
        for i in range(len(wavelengths)):
            alpha_val = alpha_gas(wavelength = wavelengths[i], gas = "CO2")
            
            calculated[i] = alpha_val

        np.testing.assert_allclose(true_values, calculated, rtol=tolerance)
      
        pass
    
    def test_kings_factor_CO2(self):

        true_values = np.array([1.150, 1.150, 1.150])

        calculated = np.nan * np.zeros(wavelengths.size)
        
        for i in range(len(wavelengths)):
            Fk_val = kings_factor(wavelength = wavelengths[i], gas = "CO2")
            
            calculated[i] = Fk_val

        np.testing.assert_allclose(true_values, calculated, atol = 3)
      
        pass

    def test_alpha_Ar(self):

        true_values = np.array([1.910131e-40, 1.863451e-40, 1.837231e-40])

        calculated = np.nan * np.zeros(wavelengths.size)
        
        for i in range(len(wavelengths)):
            alpha_val = alpha_gas(wavelength = wavelengths[i], gas = "Ar")
            
            calculated[i] = alpha_val

        np.testing.assert_allclose(true_values, calculated, rtol=tolerance)
      
        pass
    
    def test_kings_factor_Ar(self):

        true_values = np.array([0.000, 0.000, 0.000])

        calculated = np.nan * np.zeros(wavelengths.size)
        
        for i in range(len(wavelengths)):
            Fk_val = kings_factor(wavelength = wavelengths[i], gas = "Ar")
            
            calculated[i] = Fk_val

        np.testing.assert_allclose(true_values, calculated, atol = 3)
      
        pass

    def test_alpha_H2O(self):

        true_values = np.array([1.737396e-40, 1.672388e-40, 1.635035e-40])

        calculated = np.nan * np.zeros(wavelengths.size)
        
        for i in range(len(wavelengths)):
            alpha_val = alpha_gas(wavelength = wavelengths[i], gas = "H2O")
            
            calculated[i] = alpha_val

        np.testing.assert_allclose(true_values, calculated, rtol=tolerance)
      
        pass
    
    def test_kings_factor_H2O(self):

        true_values = np.array([1.001, 1.001, 1.001])

        calculated = np.nan * np.zeros(wavelengths.size)
        
        for i in range(len(wavelengths)):
            Fk_val = kings_factor(wavelength = wavelengths[i], gas = "H2O")
            
            calculated[i] = Fk_val

        np.testing.assert_allclose(true_values, calculated, atol=3)
      
        pass
class other(unittest.TestCase):

    def setUp(self):
        pass   
    
    
    def test_placzek_teller(self):

        Js = np.arange(0,100)
        
        true_values = np.ones(len(Js))

        calculated = np.nan * np.zeros(len(Js))
        
        
        X_O = placzek_teller(J = Js, branch = 'O')
        X_S = placzek_teller(J = Js, branch = 'S')
        X_Q = placzek_teller(J = Js, branch = 'Q')
            
        for i in range(len(Js)):
            calculated[i] = np.nansum([X_O[i], X_Q[i], X_S[i]])

        np.testing.assert_allclose(true_values, calculated, rtol=tolerance)
      
        pass
    
if __name__ == "__main__":
    unittest.main()
    