import unittest

import numpy as np

from algorithms.gram_schmidt import *


class TestGramSchmidt(unittest.TestCase):
    def test_gram_schmidt_single(self):
        vec = np.array([[1, 6, -5], [1, 7, 3], [1, -9, 6]])
        new_vec = gramschmidt(vec)
        print("\nnew_vec", new_vec)
        # Check if the result is orthonormal
        result = np.einsum("ij,ik->jk", new_vec, new_vec.conj())
        print("\nresult", result)
        eye = np.eye(len(new_vec))
        print("\neye", eye)
        self.assertTrue(np.allclose(result, eye))

    def test_gram_schmidt_double(self):
        vec1 = np.array([[1, 1, 1], [1, 2, 3]])
        vec2 = np.array([[1, 3, 6]])
        threshold = 0.01
        v2_new = gramschmidt2(vec1, vec2, threshold)
        result = np.einsum("ij,ik->jk", vec1, v2_new.conj())
        np.testing.assert_array_almost_equal(result, np.eye(len(result)))

    def test_gram_schmidt_complex(self):
        vec = np.array(
            [
                [1 + 1j, 2 + 2j, 3 + 3j],
                [4 + 4j, 5 + 5j, 6 + 6j],
                [7 + 7j, 8 + 8j, 9 + 9j],
            ]
        )
        new_vec = gramschmidt(vec)
        result = np.einsum("ij,ik->jk", new_vec, new_vec.conj())
        expected = np.eye(3, dtype=np.complex_)
        np.testing.assert_array_almost_equal(result, expected)


# if __name__ == '__main__':
#     unittest.main()
