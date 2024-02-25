import numpy as np


def make_sparse_matrix(n, sparse_factor, complex_valued=False):
    """
    Generate a sparse matrix with a given sparsity factor.

    Parameters:
    n (int): The size of the square matrix.
    sparse_factor (float): The sparsity factor, ranging from 0 to 1.
    complex (bool, optional): Whether to generate a complex matrix. Defaults to False.

    Returns:
    numpy.ndarray: The generated sparse matrix.
    """
    mat = np.random.rand(n, n)

    if complex_valued:
        mat = mat + 1j * np.random.rand(n, n)

    mat[mat < sparse_factor] = 0
    return mat / (1 - sparse_factor)


def is_symmetric(a, tol=1e-10):
    """
    Check if a given matrix is symmetric within a given tolerance.

    Parameters:
    a (numpy.ndarray): The input matrix.
    tol (float, optional): The tolerance within which the matrix is considered symmetric. Defaults to 1e-8.

    Returns:
    bool: True if the matrix is symmetric within the given tolerance, False otherwise.
    """
    return np.all(np.abs(a-a.T) < tol)


def np_diag(mat, n_roots=3):
    """
    Compute the eigenvalues and eigenvectors of a given matrix using numpy, and return the first 'n_roots' roots.

    Parameters:
    mat (numpy.ndarray): The input matrix.
    n_roots (int, optional): The number of roots to return. Defaults to 3.

    Returns:
    tuple: A tuple containing two numpy.ndarrays. The first array contains the eigenvalues of the matrix,
           and the second array contains the corresponding eigenvectors. Only the first 'n_roots' roots are returned.
    """
    return np.linalg.eig(mat)[0][:n_roots], np.linalg.eig(mat)[1][:, :n_roots]

