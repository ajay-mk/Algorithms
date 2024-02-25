import unittest

import numpy as np

import algorithms.utils.matrix_tools as mt
from algorithms.davidson import davidson_general

# Set matrix size and parameters
n = 9
sparse_factor = 0.3
n_roots = 3


class TestDavidson(unittest.TestCase):
    def test_davidson_symm(self):
        # Initialize a random symmetric matrix
        symm_mat = mt.make_sparse_matrix(n, sparse_factor, False)
        symm_mat += symm_mat.T
        symm_mat /= 2
        # print("Symmetric matrix:\n", symm_mat)

        eigs = davidson_general(symm_mat, n_roots, 1e-9)[0]
        # print("\nEigenvalues from Davidson:\n", E)

        np_eigs = mt.np_diag(symm_mat, n_roots)[0]
        print("\nEigenvalues from numpy:\n", np_eigs)

        condition = np.allclose(np_eigs, eigs, atol=1e-7)
        self.assertTrue(condition, "Eigenvalues do not match!")

    def test_davidson_real(self):
        # n = 50
        sparse_factor = 0.2
        n_roots = 3

        # Initialize a random real matrix
        real_mat = mt.make_sparse_matrix(n, sparse_factor, False)

        eigs = davidson_general(real_mat, n_roots, 1e-9)[0]
        # print("\nEigenvalues from Davidson:\n", E)

        np_eigs = mt.np_diag(real_mat, n_roots)[0]
        print("\nEigenvalues from numpy:\n", np_eigs)

        condition = np.allclose(np_eigs, eigs, atol=1e-7)
        self.assertTrue(condition, "Eigenvalues do not match!")

    def test_davidson_complex(self):
        # n = 50
        sparse_factor = 0.2
        n_roots = 3

        # Initialize a random real matrix
        real_mat = mt.make_sparse_matrix(n, sparse_factor, True)

        eigs = davidson_general(real_mat, n_roots, 1e-9)[0]
        # print("\nEigenvalues from Davidson:\n", E)

        np_eigs = mt.np_diag(real_mat, n_roots)[0]
        # print("\nEigenvalues from numpy:\n", np_eigs)

        condition = np.allclose(np_eigs, eigs, atol=1e-7)
        self.assertTrue(condition, "Eigenvalues do not match!")


# if __name__ == "__main__":
#     unittest.main()
#     TestDavidson.test_davidson_real()
#     TestDavidson.test_davidson_symm()
#     TestDavidson.test_davidson_complex()
