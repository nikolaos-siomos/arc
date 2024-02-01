import unittest

from lidar_molecular.rayleigh_scattering_bucholtz import *
from lidar_molecular.utilities import number_density_at_pt


class BucholtzTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_number_density_in_STP_ideal(self):
        temperature = 288.15  # in K
        pressure = 1013.25  # in hPa
        relative_humidity = 0
        density_stp = number_density_at_pt(pressure, temperature, relative_humidity, ideal=True)
        self.assertAlmostEqual(density_stp, 2.546917378744801e+25, 12)

    def test_depolariation(self):
        rho_200 = depolarization_factor(200)
        self.assertAlmostEqual(rho_200, 4.545 * 1e-2, 15)

        rho_350 = depolarization_factor(350)
        self.assertAlmostEqual(rho_350, 3.010 * 1e-2, 15)

        rho_1000 = depolarization_factor(1000)
        self.assertAlmostEqual(rho_1000, 2.730 * 1e-2, 15)

    def test_kigs_factor(self):
        fk_200 = king_correction_factor(200)
        self.assertEqual(round(fk_200, 3), 1.080)

        fk_350 = king_correction_factor(350)
        self.assertEqual(round(fk_350, 3), 1.052)

        fk_1000 = king_correction_factor(1000)
        self.assertEqual(round(fk_1000, 3), 1.047)

    def test_scattering_cross_section(self):
        sigma_200 = scattering_cross_section(200)
        self.assertEqual(round(sigma_200 * 1e25, 3), 3.612)

        # Check the refractive index for wavelengths.

        sigma_350 = scattering_cross_section(350)
        self.assertEqual(round(sigma_350 * 1e26, 3), 2.924)

        sigma_4000 = scattering_cross_section(4000)
        self.assertEqual(round(sigma_4000 * 1e30, 3), 1.550)


if __name__ == "__main__":
    unittest.main()
