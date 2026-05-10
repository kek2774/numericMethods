import numpy as np
from .methods_utils import norm


def solve_mpi_matrix(
    eps: float, A: np.ndarray, b: np.ndarray, maxiter: int = 10000
) -> tuple[np.ndarray, np.ndarray] | None:

    # Чтобы не портить исходные данные
    A_matrix: np.ndarray = A.copy().astype(float)
    b_vec: np.ndarray = b.copy().astype(float)

    dim: int = A_matrix.shape[0]

    x_vec_old: np.ndarray = np.zeros(dim, dtype=float)
    x_vec_new: np.ndarray = np.zeros(dim, dtype=float)
    residual_vec: np.ndarray = np.zeros(dim, dtype=float)

    # Матрица перехода
    B_matrix: np.ndarray = A.copy().astype(float)
    for i in range(dim):
        if abs(A_matrix[i, i]) < 1e-12:
            return None
        B_matrix[i, i] = 0
        B_matrix[i] /= -A_matrix[i, i]
        b_vec[i] /= A_matrix[i, i]

    # Считаем все нормы
    norms: np.ndarray = np.array(
        [
            norm.norm_1(B_matrix),
            norm.norm_inf(B_matrix),
            norm.norm_fro(B_matrix),
        ],
        dtype=float,
    )
    vec_norm = norm.vec_norm_1
    correct_norm = None

    # Проверка на достаточное условие сходимости
    for i in range(len(norms)):
        if norms[i] < 1 and i == 0:
            correct_norm = norms[i]
            vec_norm = norm.vec_norm_1
        if norms[i] < 1 and i == 1:
            correct_norm = norms[i]
            vec_norm = norm.vec_norm_inf
        if norms[i] < 1 and i == 2:
            correct_norm = norms[i]
            vec_norm = norm.vec_norm_fro

    if correct_norm is None:
        return None

    # Начальное приближение
    x_vec_old = b_vec.copy()

    # Итерации МПИ
    iterations = 0
    while True:
        x_vec_new = b_vec.copy()
        for i in range(dim):
            for j in range(dim):
                if i != j:
                    x_vec_new[i] += B_matrix[i, j] * x_vec_old[j]
        iterations += 1

        x_diff_norm = vec_norm(x_vec_new - x_vec_old)
        if correct_norm / (1 - correct_norm) * x_diff_norm < eps:
            break

        if iterations >= maxiter:
            return None

        x_vec_old = x_vec_new.copy()

    # Невязки
    for i in range(dim):
        summa_for_residual: float = 0
        for j in range(dim):
            summa_for_residual += A[i, j] * x_vec_new[j]
        residual_vec[i] = b[i] - summa_for_residual

    return (x_vec_new, residual_vec)
