import numpy as np

import algorithms.utils.matrix_tools as mt


def davidson_general(matrix, n_roots, tol, max_iter):
    assert n_roots > 0, "Number of roots must be greater than 0"
    assert matrix.shape[0] == matrix.shape[1], "Matrix must be square"

    dim = matrix.shape[0]
    dim_subspace = 2 * n_roots
    guess_vectors = np.eye(dim, dim_subspace, dtype=np.complex128)

    itr = 0
    converged = False

    while not converged and itr < max_iter:
        itr += 1
        if itr == 1:
            print("Starting Davidson algorithm iterations\n")

        # Project matrix onto subspace
        subspace = np.einsum("ij,jk->ik", matrix, guess_vectors)
        subspace = np.einsum("ij,ik->jk", subspace, guess_vectors.conj())

        # Diagonalize subspace
        # note: always use np.linalg.eig, not np.linalg.eigh
        eigenvalues, subspace_eigenvectors = np.linalg.eig(subspace)

        # Trim results
        eigenvalues, subspace_eigenvectors = (
            eigenvalues[:n_roots],
            subspace_eigenvectors[:, :n_roots],
        )
