"""
Unit test of the refractive_index functions.

The test is performed using the tabular values found in the referenced paper.
"""

import os
import unittest
from lidar_molecular.refractive_index import *

# Get the data path
#current_path = os.path.dirname(__file__)
#data_base_path = os.path.join(current_path, '../data/bucholtz_tabular_values/')

class TestRefractiveIndexN2(unittest.TestCase):
    def test_raises_value_error_if_wrong_method(self):
        wavelength = 500  # Not important here
        method = 'wrong_name'
        self.assertRaises(ValueError, n_N2, wavelength, method=method)

    def test_griesmann_burnett_no_extrapolation(self):
        wavelengths = np.array([144, 146.3, 158.8, 171.3, 181.3, 212.5, 228.8, 255.0, 268.8, 271.])
        true_values = np.array([np.nan, 1.0004042484761, 1.0003805900101, 1.0003642819517, 1.0003545434982, 1.0003348442346, 1.0003283159731, 1.0003207825974, 1.0003178062198 ,np.nan])  

        calculated, _, _ = n_N2(wavelengths, temperature=273.15, pressure=1013.25, method='griesmann_burnett', extrapolate=False)

        np.testing.assert_allclose(true_values, calculated, rtol=1e-12)

    def test_griesmann_burnett_with_extrapolation(self):
        wavelengths = np.array([144, 146.3, 158.8, 171.3, 181.3, 212.5, 228.8, 255.0, 268.8, 271.])
        true_values = np.array([1.000409789154, 1.0004042484761, 1.0003805900101, 1.0003642819517, 1.0003545434982, 1.0003348442346, 1.0003283159731, 1.0003207825974, 1.0003178062198 ,1.000317379599])  

        calculated, _, _ = n_N2(wavelengths, temperature=273.15, pressure=1013.25, method='griesmann_burnett', extrapolate=True)

        np.testing.assert_allclose(true_values, calculated, rtol=1e-12)
        
    def test_boerzsoenyi_no_extrapolation(self):
        wavelengths = np.array([399, 406, 508, 604, 712, 814, 910, 994, 1001.])

        true_values = np.array([np.nan, 1.0002997820217, 1.0002960938694, 1.0002942634855, 1.0002930514072, 1.0002923313114, 1.0002918668418, 1.0002915676087, np.nan])  

        calculated, _, _ = n_N2(wavelengths, temperature=273.00, pressure=1000.00, method='boerzsoenyi', extrapolate=False)

        np.testing.assert_allclose(true_values, calculated, rtol=1e-12)

    def test_boerzsoenyi_with_extrapolation(self):
        wavelengths = np.array([399, 406, 508, 604, 712, 814, 910, 994, 1001.])

        true_values = np.array([1.000300154588, 1.0002997820217, 1.0002960938694, 1.0002942634855, 1.0002930514072, 1.0002923313114, 1.0002918668418, 1.0002915676087, 1.000291546069])  

        calculated, _, _ = n_N2(wavelengths, temperature=273.00, pressure=1000.00, method='boerzsoenyi', extrapolate=True)

        np.testing.assert_allclose(true_values, calculated, rtol=1e-12)
        
    def test_peck_khanna_no_extrapolation(self):
        wavelengths = np.array([466, 483.8, 722.4, 992.9, 1327., 1677., 2043., 2059.])
        true_values = np.array([np.nan, 1.0003006576234, 1.0002968086606, 1.0002953689610, 1.0002946629329, 1.0002943287434, 1.0002941465801, np.nan])  

        calculated, _, _ = n_N2(wavelengths, temperature=273.15, pressure=1013.25, method='peck_khanna', extrapolate=False)

        np.testing.assert_allclose(true_values, calculated, rtol=1e-12)

    def test_peck_khanna_with_extrapolation(self):
        wavelengths = np.array([466, 483.8, 722.4, 992.9, 1327., 1677., 2043., 2059.])
        true_values = np.array([1.000301211467, 1.0003006576234, 1.0002968086606, 1.0002953689610, 1.0002946629329, 1.0002943287434, 1.0002941465801, 1.000294140760])  
        
        calculated, _, _ = n_N2(wavelengths, temperature=273.15, pressure=1013.25, method='peck_khanna', extrapolate=True)

        np.testing.assert_allclose(true_values, calculated, rtol=1e-12)
        
    def test_combined_no_extrapolation(self):

        wavelengths = np.array([144, 146.3, 158.8, 171.3, 181.3, 212.5, 228.8, 255.0, 268.8, 350, 406, 514, 604, 706, 814, 916, 994, 1000., 1327., 1677., 2043., 2059.])

        true_values = np.array([np.nan, 1.000376667109, 1.000354622926, 1.000339427605, 1.000330353627, 1.000311998479, 1.000305915651, 1.000298896287, 1.000296122991, 1.000286392933, 1.000282874448, 1.000279255553, 1.00027766717 , 1.000276573123, 1.000275843974, 1.000275382877, 1.000275123346, 1.000275105898, 1.000274558793, 1.000274247406,  1.000274077671, np.nan])

        calculated, _, _ = n_N2(wavelengths, temperature=293.15, pressure=1013.25, method='combined', extrapolate=False)

        np.testing.assert_allclose(true_values, calculated, rtol=1e-12)
        	
    def test_combined_with_extrapolation(self):

        wavelengths = np.array([144, 146.3, 158.8, 171.3, 181.3, 212.5, 228.8, 255.0, 268.8, 350, 406, 514, 604, 706, 814, 916, 994, 1000., 1327., 1677., 2043., 2059.])

        true_values = np.array([1.000381829729, 1.000376667109, 1.000354622926, 1.000339427605, 1.000330353627, 1.000311998479, 1.000305915651, 1.000298896287, 1.000296122991, 1.000286392933, 1.000282874448, 1.000279255553, 1.00027766717 , 1.000276573123, 1.000275843974, 1.000275382877, 1.000275123346, 1.000275105898, 1.000274558793, 1.000274247406, 1.000274077671, 1.000274072248])
        
        calculated, _, _ = n_N2(wavelengths, temperature=293.15, pressure=1013.25, method='combined', extrapolate=True)

        np.testing.assert_allclose(true_values, calculated, rtol=1e-12)


class TestRefractiveIndexO2(unittest.TestCase):
    def test_raises_value_error_if_wrong_method(self):
        wavelength = 500  # Not important here
        method = 'wrong_name'
        self.assertRaises(ValueError, n_O2, wavelength, method=method)

    def test_smith_no_extrapolation(self):
        wavelengths = np.array([183, 191.85, 198.64, 243.59, 263.21, 289.])
        true_values = np.array([np.nan, 1.00036120, 1.00034670, 1.00030650, 1.00029860, np.nan])  

        calculated, _, _ = n_O2(wavelengths, temperature=273.15, pressure=1013.25, method='smith', extrapolate=False)

        np.testing.assert_allclose(true_values, calculated, rtol=1e-8)

    def test_smith_with_extrapolation(self):
        wavelengths = np.array([183, 191.85, 198.64, 243.59, 263.21, 289.])
        true_values = np.array([1.00038229, 1.00036120, 1.00034670, 1.00030650, 1.00029860, 1.000291387])  

        calculated, _, _ = n_O2(wavelengths, temperature=273.15, pressure=1013.25, method='smith', extrapolate=True)

        np.testing.assert_allclose(true_values, calculated, rtol=1e-8)

    def test_zhang_no_extrapolation(self):
        wavelengths = np.array([399, 414., 596., 806., 1030., 1324., 1548., 1772., 1801.])
        true_values = np.array([np.nan, 1.0002577145995, 1.0002519092156,1.0002495986565,1.0002485451762, 1.0002478967003, 1.0002476316967, 1.0002474612478, np.nan])  

        calculated, _, _ = n_O2(wavelengths, temperature=293.15, pressure=1013.25, method='zhang', extrapolate=False)

        np.testing.assert_allclose(true_values, calculated, rtol=1e-12)

    def test_zhang_with_extrapolation(self):
        wavelengths = np.array([399, 414., 596., 806., 1030., 1324., 1548., 1772., 1801.])
        true_values = np.array([1.000258617037, 1.0002577145995, 1.0002519092156,1.0002495986565,1.0002485451762, 1.0002478967003, 1.0002476316967, 1.0002474612478, 1.000247443728])  

        calculated, _, _ = n_O2(wavelengths, temperature=293.15, pressure=1013.25, method='zhang', extrapolate=True)

        np.testing.assert_allclose(true_values, calculated, rtol=1e-12)
        
    def test_combined_no_extrapolation(self):
        
        wavelengths = np.array([183, 191.85, 198.64, 243.59, 263.21, 289., 350, 414., 596., 806., 1030., 1324., 1548., 1772., 1801.])

        true_values = np.array([np.nan, 1.000336555942, 1.000323045305, 1.000285588207, 1.000278227231, 1.000271222013, 1.000262548604, 1.0002577146, 1.000251909216, 1.000249598656, 1.000248545176, 1.0002478967, 1.000247631697, 1.000247461248, np.nan])

        calculated, _, _ = n_O2(wavelengths, temperature=293.15, pressure=1013.25, method='combined', extrapolate=False)

        np.testing.assert_allclose(true_values, calculated, rtol=1e-8)
        
        	
    def test_combined_with_extrapolation(self):

        wavelengths = np.array([183, 191.85, 198.64, 243.59, 263.21, 289., 350, 414., 596., 806., 1030., 1324., 1548., 1772., 1801.])

        true_values = np.array([1.000356215622, 1.000336555942, 1.000323045305, 1.000285588207, 1.000278227231, 1.000271222013, 1.000262548604, 1.0002577146, 1.000251909216, 1.000249598656, 1.000248545176, 1.0002478967, 1.000247631697, 1.000247461248, 1.000247443728])
        
        calculated, _, _ = n_O2(wavelengths, temperature=293.15, pressure=1013.25, method='combined', extrapolate=True)

        np.testing.assert_allclose(true_values, calculated, rtol=1e-8)
          	
        
class TestRefractiveIndexAr(unittest.TestCase):
    def test_raises_value_error_if_wrong_method(self):
        wavelength = 500  # Not important here
        method = 'wrong_name'
        self.assertRaises(ValueError, n_Ar, wavelength, method=method)

    def test_bideau_mehu_larsen_no_extrapolation(self):
        wavelengths = np.array([140., 144.7, 200.2, 307.0, 354.1, 600.])
        true_values = np.array([np.nan, 1.000384471176, 1.0003220826967, 1.0002943782739, 1.000289893794, np.nan])
        
        calculated, _, _ = n_Ar(wavelengths, temperature=273.15, pressure=1013.25, method='bideau_mehu_larsen', extrapolate=False)

        np.testing.assert_allclose(true_values, calculated, rtol=1e-12)

    def test_bideau_mehu_larsen_with_extrapolation(self):
        wavelengths = np.array([140., 144.7, 200.2, 307.0, 354.1, 600.])
        true_values = np.array([1.000396573279, 1.000384471176, 1.0003220826967, 1.0002943782739, 1.000289893794, 1.000281499504])
        
        calculated, _, _ = n_Ar(wavelengths, temperature=273.15, pressure=1013.25, method='bideau_mehu_larsen', extrapolate=True)

        np.testing.assert_allclose(true_values, calculated, rtol=1e-12)

    def test_boerzsoenyi_no_extrapolation(self):
        wavelengths = np.array([399., 412., 580., 682., 778., 886., 994., 1001.])
        true_values = np.array([np.nan, 1.0002826321595, 1.0002782092196, 1.0002770002479, 1.0002762764288, 1.0002757303986, 1.0002753544109, np.nan])
        
        calculated, _, _ = n_Ar(wavelengths, temperature=273.00, pressure=1000.00, method='boerzsoenyi', extrapolate=False)

        np.testing.assert_allclose(true_values, calculated, rtol=1e-12)    
        
    def test_boerzsoenyi_with_extrapolation(self):
        wavelengths = np.array([399., 412., 580., 682., 778., 886., 994., 1001.])
        true_values = np.array([1.000283239795, 1.0002826321595, 1.0002782092196, 1.0002770002479, 1.0002762764288, 1.0002757303986, 1.0002753544109, 1.000275334196])
        
        calculated, _, _ = n_Ar(wavelengths, temperature=273.00, pressure=1000.00, method='boerzsoenyi', extrapolate=True)

        np.testing.assert_allclose(true_values, calculated, rtol=1e-12)


    def test_peck_fisher_no_extrapolation(self):
        wavelengths = np.array([467., 483.8, 802.0, 1136., 1375., 1709., 2043., 2060.])
        true_values = np.array([np.nan,  1.0002838796938, 1.0002797584877, 1.0002786047975, 1.0002782435036, 1.0002779703293, 1.0002778200905, np.nan])
        
        calculated, _, _ = n_Ar(wavelengths, temperature=273.15, pressure=1013.25, method='peck_fisher', extrapolate=False)

        np.testing.assert_allclose(true_values, calculated, rtol=1e-12)
        
        	
    def test_peck_fisher_with_extrapolation(self):
        wavelengths = np.array([467., 483.8, 802.0, 1136., 1375., 1709., 2043., 2060.])
        true_values = np.array([1.000284364539,  1.0002838796938, 1.0002797584877, 1.0002786047975, 1.0002782435036, 1.0002779703293, 1.0002778200905, 1.000277814339])
        
        calculated, _, _ = n_Ar(wavelengths, temperature=273.15, pressure=1013.25, method='peck_fisher', extrapolate=True)

        np.testing.assert_allclose(true_values, calculated, rtol=1e-12)
        
    def test_combined_no_extrapolation(self):

        wavelengths = np.array([140., 144.7, 200.2, 307.0, 354.1, 412., 580., 682., 778., 886., 994., 483.8, 802.0, 1136., 1375., 1709., 2043., 2060.])

        true_values = np.array([np.nan, 1.00035823927 , 1.000300107679, 1.000274293557, 1.000270115056, 1.000266691874, 1.000262518396, 1.000261377613, 1.000260694618, 1.000260179385, 1.000259824604, 1.000264352156, 1.000260561636, 1.000259596313, 1.00025925967 , 1.000259005135, 1.000258865147, np.nan])
        
        calculated, _, _ = n_Ar(wavelengths, temperature=293.15, pressure=1013.25, method='combined', extrapolate=False)

        np.testing.assert_allclose(true_values, calculated, rtol=1e-12)
            	
    def test_combined_with_extrapolation(self):

        wavelengths = np.array([140., 144.7, 200.2, 307.0, 354.1, 412., 580., 682., 778., 886., 994., 483.8, 802.0, 1136., 1375., 1709., 2043., 2060.])

        true_values = np.array([1.000369515614, 1.00035823927 , 1.000300107679, 1.000274293557, 1.000270115056, 1.000266691874, 1.000262518396, 1.000261377613, 1.000260694618, 1.000260179385, 1.000259824604, 1.000264352156, 1.000260561636, 1.000259596313, 1.00025925967 , 1.000259005135, 1.000258865147, 1.000258859788])
        
        calculated, _, _ = n_Ar(wavelengths, temperature=293.15, pressure=1013.25, method='combined', extrapolate=True)

        np.testing.assert_allclose(true_values, calculated, rtol=1e-12)
  
        	
class TestRefractiveIndexCO2(unittest.TestCase):
    def test_raises_value_error_if_wrong_method(self):
        wavelength = 500  # Not important here
        method = 'wrong_name'
        self.assertRaises(ValueError, n_CO2, wavelength, method=method)

    def test_bideau_mehu_no_extrapolation(self):
        wavelengths = np.array([180., 195.8, 407.8, 604.6, 755.9, 952.7, 1165., 1377., 1679., 1700.])
        true_values = np.array([np.nan, 1.0005307566073, 1.0004584950913, 1.0004484246442, 1.0004452544798, 1.0004428606430, 1.0004410667500, 1.0004395075400, 1.0004371952107, np.nan])
        
        calculated, _, _ = n_CO2(wavelengths, temperature=273.15, pressure=1013.25, method='bideau_mehu', extrapolate=False)

        np.testing.assert_allclose(true_values, calculated, rtol=1e-12)
        
    def test_bideau_mehu_with_extrapolation(self):
        wavelengths = np.array([180., 195.8, 407.8, 604.6, 755.9, 952.7, 1165., 1377., 1679., 1700.])
        true_values = np.array([1.000552270823, 1.0005307566073, 1.0004584950913, 1.0004484246442, 1.0004452544798, 1.0004428606430, 1.0004410667500, 1.0004395075400, 1.0004371952107, 1.000437020524])
        
        calculated, _, _ = n_CO2(wavelengths, temperature=273.15, pressure=1013.25, method='bideau_mehu', extrapolate=True)

        np.testing.assert_allclose(true_values, calculated, rtol=1e-12)

    def test_old_no_extrapolation(self):
        wavelengths = np.array([479., 493.5, 640.6, 814.4, 988.2, 1175., 1363., 1563., 1804., 1818.])
        true_values = np.array([np.nan, 1.0004527308977, 1.0004477604878, 1.0004447854840, 1.0004429354474, 1.0004414051984, 1.0004400048783, 1.0004384754476, 1.0004363957421, np.nan])
        
        calculated, _, _ = n_CO2(wavelengths, temperature=273.15, pressure=1013.25, method='old', extrapolate=False)

        np.testing.assert_allclose(true_values, calculated, rtol=1e-12)
        
    def test_old_with_extrapolation(self):
        wavelengths = np.array([479., 493.5, 640.6, 814.4, 988.2, 1175., 1363., 1563., 1804., 1818.])
        true_values = np.array([1.000453476494, 1.0004527308977, 1.0004477604878, 1.0004447854840, 1.0004429354474, 1.0004414051984, 1.0004400048783, 1.0004384754476, 1.0004363957421, 1.000436263195])
        
        calculated, _, _ = n_CO2(wavelengths, temperature=273.15, pressure=1013.25, method='old', extrapolate=True)

        np.testing.assert_allclose(true_values, calculated, rtol=1e-12)

    def test_combined_no_extrapolation(self):

        wavelengths = np.array([180., 195.8, 407.8, 604.6, 755.9, 952.7, 1165., 1377., 1563., 1804., 1818.])

        true_values = np.array([np.nan, 1.000494543036, 1.000427212282, 1.000417828982, 1.000414875131, 1.000412644635, 1.000410973146, 1.000409520327, 1.00040822805 , 1.000406620859, np.nan])
        
        calculated, _, _ = n_CO2(wavelengths, temperature=293.15, pressure=1013.25, method='combined', extrapolate=False)

        np.testing.assert_allclose(true_values, calculated, rtol=1e-12)
            	
    def test_combined_with_extrapolation(self):

        wavelengths = np.array([180., 195.8, 407.8, 604.6, 755.9, 952.7, 1165., 1377., 1563., 1804., 1818.])

        true_values = np.array([1.000514589209, 1.000494543036, 1.000427212282, 1.000417828982, 1.000414875131, 1.000412644635, 1.000410973146, 1.000409520327, 1.00040822805 , 1.000406620859, 1.000406497356])
        
        calculated, _, _ = n_CO2(wavelengths, temperature=293.15, pressure=1013.25, method='combined', extrapolate=True)

        np.testing.assert_allclose(true_values, calculated, rtol=1e-12)


class TestRefractiveIndexH2O(unittest.TestCase):
    def test_raises_value_error_if_wrong_method(self):
        wavelength = 500  # Not important here
        method = 'wrong_name'
        self.assertRaises(ValueError, n_H2O, wavelength, method=method)

    def test_cidor_no_extrapolation(self):
        wavelengths = np.array([349., 644.0, 587.7, 546.2, 608.7, 480.1, 467.9, 447.2, 435.9, 1201.])
        true_values = np.array([np.nan, 1.000007010, 1.000007090, 1.000007112, 1.000007138, 1.000007166, 1.000007186, 1.000007196, 1.000007229, np.nan])
        
        calculated, _, _ = n_H2O(wavelengths, temperature=294.05, pressure=32.19, method='cidor', extrapolate=False)

        np.testing.assert_allclose(true_values, calculated, rtol=1e-6)
        
    def test_cidor_with_extrapolation(self):
        wavelengths = np.array([349., 644.0, 587.7, 546.2, 608.7, 480.1, 467.9, 447.2, 435.9, 1201.])
        true_values = np.array([1.000007798, 1.000007010, 1.000007090, 1.000007112, 1.000007138, 1.000007166, 1.000007186, 1.000007196, 1.000007229, 1.000007308])
        
        calculated, _, _ = n_H2O(wavelengths, temperature=294.05, pressure=32.19, method='cidor', extrapolate=True)

        np.testing.assert_allclose(true_values, calculated, rtol=1e-6)
          

if __name__ == "__main__":
    unittest.main()
    