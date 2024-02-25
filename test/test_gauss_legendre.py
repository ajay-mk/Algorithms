import numpy as np
import unittest
import sys

sys.path.append('Gauss_Legendre_Quadrature')

from gauss_legendre import gauss_legendre

class TestGaussLegendre(unittest.TestCase):
    def test_gauss_legendre_2(self):
        x, w = gauss_legendre(2, 0, 1)
        expected_x = np.array([0.21132487, 0.78867513])
        expected_w = np.array([0.50000000, 0.500000])
        self.assertTrue(np.allclose(x, expected_x))
        self.assertTrue(np.allclose(w, expected_w))

    def test_gauss_legendre_5(self):
        x, w = gauss_legendre(5, 1, 3)
        expected_x = np.array([1.09382015, 1.46153069, 2.0, 2.53846931, 2.90617985])
        expected_w = np.array([0.23692689, 0.47862867, 0.56888889, 0.47862867, 0.23692689])
        self.assertTrue(np.allclose(x, expected_x))
        self.assertTrue(np.allclose(w, expected_w))

if __name__ == '__main__':
    unittest.main()