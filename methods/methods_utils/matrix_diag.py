from unittest import result

import numpy as np


def get_down_diag_matrix(A: np.ndarray) -> np.ndarray:
    dim1, dim2 = A.shape

    result_matrix: np.ndarray = np.zeros(A.shape, dtype=float)
    for i in range(dim1):
        for j in range(dim2):
            if j <= i:
                result_matrix[i, j] = A[i, j]

    return result_matrix


def get_upp_diag_matrix(A: np.ndarray) -> np.ndarray:
    dim1, dim2 = A.shape

    result_matrix: np.ndarray = np.zeros(A.shape, dtype=float)
    for i in range(dim1):
        for j in range(dim2):
            if j > i:
                result_matrix[i, j] = A[i, j]

    return result_matrix


def get_diag_matrix(A: np.ndarray) -> np.ndarray:
    dim1, _ = A.shape

    result_matrix: np.ndarray = np.zeros(A.shape, dtype=float)
    for i in range(dim1):
        result_matrix[i, i] = A[i, i]

    return result_matrix


def get_diag_as_vector(A: np.ndarray) -> np.ndarray:
    dim = A.shape[0]
    result_vector: np.ndarray = np.zeros(dim, dtype=float)

    for i in range(dim):
        for j in range(dim):
            if i == j:
                result_vector[i] = A[i, j]

    return result_vector
