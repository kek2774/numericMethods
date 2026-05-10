import numpy as np


def solve_gauss(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    # Чтобы не портить исходный массив
    A_matrix: np.ndarray = A.copy().astype(float)
    b_vec: np.ndarray = b.copy().astype(float)

    dim: int = A_matrix.shape[0]
    x_vec: np.ndarray = np.zeros(dim, dtype=float)
    residual_vec: np.ndarray = np.zeros(dim, dtype=float)

    tmp_eps: float = 1e-9
    # Прямой ход
    for row in range(dim):
        if abs(A_matrix[row, row]) < tmp_eps:  # Если опорный элемент 0
            for i in range(row + 1, dim):
                if abs(A_matrix[i, row]) >= tmp_eps:
                    # Меняем местами строки, если нашли ненулевой опорный элемент
                    A_matrix[[row, i]] = A_matrix[[i, row]]
                    b_vec[row], b_vec[i] = b_vec[i], b_vec[row]
                    break

        # Если не нашелся ненулевой опорный элемент
        if abs(A_matrix[row, row]) < tmp_eps:
            return None

        # Прямой ход
        pivot = A_matrix[row, row]
        A_matrix[row] /= pivot
        b_vec[row] /= pivot
        for j in range(row + 1, dim):
            factor = A_matrix[j, row]
            A_matrix[j] -= A_matrix[row] * factor
            b_vec[j] -= b_vec[row] * factor

    # Обратный ход
    for row in range(dim - 1, -1, -1):
        summa: float | int = 0
        for i in range(row + 1, dim):
            summa += A_matrix[row, i] * x_vec[i]

        x_vec[row] = b_vec[row] - summa

    # Вычисление невязок
    for row in range(dim):
        summa_for_residual: float | int = 0
        for i in range(dim):
            summa_for_residual += A[row, i] * x_vec[i]
        residual_vec[row] = b[row] - summa_for_residual

    return (x_vec, residual_vec)
