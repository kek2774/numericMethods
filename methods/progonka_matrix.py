import numpy as np


def solve_progonka_matrix(
    A: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    tmp_eps = 1e-9
    dim = A.shape[0]

    # Чтобы не портить исходные данные
    A_matrix = A.copy()
    d_coeff = b.copy()

    x_vec = np.zeros(dim, dtype=float)
    residual_vec = np.zeros(dim, dtype=float)

    # Заполняем массивы a, b, c
    a_coeff = np.zeros(dim, dtype=float)
    b_coeff = np.zeros(dim, dtype=float)
    c_coeff = np.zeros(dim, dtype=float)
    for i in range(dim):
        b_coeff[i] = A_matrix[i, i]

        if i > 0:
            a_coeff[i] = A_matrix[i, i - 1]

        if i < dim - 1:
            c_coeff[i] = A_matrix[i, i + 1]

        if abs(b_coeff[i]) < abs(a_coeff[i]) + abs(c_coeff[i]):
            return None

    # Вычисление коэффициентов A, B

    if abs(b_coeff[0]) < tmp_eps:
        return None

    A_coeff = np.zeros(dim, dtype=float)
    B_coeff = np.zeros(dim, dtype=float)
    A_coeff[0] = -c_coeff[0] / b_coeff[0]
    B_coeff[0] = d_coeff[0] / b_coeff[0]

    for i in range(1, dim):
        A_coeff[i] = -c_coeff[i] / (b_coeff[i] + a_coeff[i] * A_coeff[i - 1])
        B_coeff[i] = (d_coeff[i] - a_coeff[i] * B_coeff[i - 1]) / (
            b_coeff[i] + a_coeff[i] * A_coeff[i - 1]
        )

    # Вычисление решений
    x_vec[-1] = B_coeff[-1]
    for i in range(dim - 2, -1, -1):
        x_vec[i] = A_coeff[i] * x_vec[i + 1] + B_coeff[i]

    # Вычисление невязок
    for i in range(dim):
        summa = 0
        for j in range(dim):
            summa += A_matrix[i, j] * x_vec[j]
        residual_vec[i] = b[i] - summa

    return (x_vec, residual_vec)
