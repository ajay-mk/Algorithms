import numpy as np


def gramschmidt(V):
    n, k = V.shape
    U = np.zeros((n, k), dtype=np.complex_)
    U[:, 0] = V[:, 0] / np.linalg.norm(V[:, 0])
    for i in range(1, k):
        U[:, i] = V[:, i]
        for j in range(i):
            # Compute the projection of V[:,i] onto U[:,j] with conjugation
            proj = np.einsum("i,i->", (U[:, j]).conj(), U[:, i]) * U[:, j]
            U[:, i] -= proj
        # Normalize the vector
        U[:, i] = U[:, i] / np.linalg.norm(U[:, i])
    return U


def gramschmidt2(V1, V2, threshold=1e-8):
    V1 = np.array(V1, dtype=np.complex_)  # Ensure complex type if needed
    V2 = np.array(V2, dtype=np.complex_)
    k1, k2 = V1.shape[0], V2.shape[0]
    n_neglect = 0

    i = 0
    while i < k2:
        # Orthogonalize against V1
        for j in range(k1):
            # Using einsum for conjugate dot product
            tmp = np.einsum("i,i->", np.conj(V1[j]), V2[i])
            V2[i] -= tmp * V1[j]

        # Orthogonalize against the rest of V2
        for k in range(i):
            # Using einsum for conjugate dot product
            tmp = np.einsum("i,i->", np.conj(V2[k]), V2[i])
            V2[i] -= tmp * V2[k]

        norm = np.linalg.norm(V2[i])
        if norm < threshold:
            print(f"Gram Schmidt neglect {i + n_neglect}th vector with norm: {norm}")
            V2 = np.delete(V2, i, axis=0)
            n_neglect += 1
            k2 -= 1
        else:
            V2[i] /= norm
            i += 1

    return V2


# def gram_schmidt(vec, threshold=1e-8, start=0):
#     """
#     Performs the Gram-Schmidt process on a given set of vectors.
#     https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process#Algorithm
#
#     Parameters:
#     vec (numpy.ndarray): The input vectors to be orthonormalized.
#     threshold (float): The threshold below which vectors are considered to be negligible and are removed.
#     start (int, optional): The index from which to start the process. Defaults to 0.
#
#     Returns:
#     vec (numpy.ndarray): The orthonormalized vectors.
#     """
#
#     k = len(vec)
#     n_neglect = 0
#
#     assert start < k, "Start index is out of bounds."
#
#     for i in range(start, k):
#         for j in range(i):
#             tmp = np.einsum('i,i->', vec[j].conj(), vec[i]) # Inner product
#             print("tmp", tmp)
#             vec[i] -= tmp * vec[j]  # axpy operation
#
#         norm = np.linalg.norm(vec[i], ord=2)
#
#         if norm < threshold:
#             print(f"Gram Schmidt neglect {i + n_neglect}th vector with norm: {norm}")
#             vec = np.delete(vec, i, 0)  # Remove the vector
#             n_neglect += 1
#             i -= 1
#             k -= 1
#         else:
#             vec[i] = vec[i] / norm  # Normalize the vector
#
#     return vec
#
#
# def gram_schmidt2(vec1, vec2, threshold):
#     """
#     Performs the Gram-Schmidt process on two given sets of vectors.
#     https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process#Algorithm
#
#     This function first orthogonalizes vec2 against vec1, and then orthogonalizes vec2 against itself.
#
#     Parameters:
#     vec1 (numpy.ndarray): The first set of input vectors.
#     vec2 (numpy.ndarray): The second set of input vectors to be orthonormalized.
#     threshold (float): The threshold below which vectors are considered to be negligible and are removed.
#
#     Returns:
#     vec2 (numpy.ndarray): The orthonormalized vectors.
#
#     """
#
#     k1 = len(vec1)
#     k2 = len(vec2)
#     n_neglect = 0
#
#     i = 0
#     while i < k2:
#         # Orthogonalize against vectors in V1
#         for j in range(k1):
#             tmp = np.einsum('i,i->', vec1[j], vec2[i])  # Inner product with V1[j]
#             vec2[i] -= tmp * vec1[j]  # Subtract projection on V1[j]
#
#         # Orthogonalize against previous vectors in V2
#         for k in range(i):
#             tmp = np.einsum('i,i->', vec2[k], vec2[i])  # Inner product with V2[k]
#             vec2[i] -= tmp * vec2[k]  # Subtract projection on V2[k]
#
#         norm = np.linalg.norm(vec2[i])
#
#         if norm < threshold:
#             print(f"Gram Schmidt neglect {i + n_neglect}th vector with norm: {norm}")
#             vec2 = np.delete(vec2, i, 0)  # Remove the vector
#             n_neglect += 1
#             k2 -= 1
#         else:
#             vec2[i] /= norm  # Normalize the vector
#             i += 1
#
#     return vec2
