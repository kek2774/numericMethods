import numpy as np


# Сумма по столбцам
def norm_1(A: np.ndarray) -> float | None:
    dim1, dim2 = A.shape
    # Проверка на квадратную матрицу
    if dim1 != dim2:
        return None

    col_sums: np.ndarray = np.zeros(dim1, dtype=float)
    for i in range(dim1):
        col_sum: float = 0
        for j in range(dim1):
            col_sum += abs(A[j, i])
        col_sums[i] = col_sum

    return float(max(col_sums))


def vec_norm_1(A: np.ndarray) -> float:
    return float(max(abs(A)))


# Сумма по строкам
def norm_inf(A: np.ndarray) -> float | None:
    dim1, dim2 = A.shape
    # Проверка на квадратную матрицу
    if dim1 != dim2:
        return None

    row_sums: np.ndarray = np.zeros(dim1, dtype=float)
    for i in range(dim1):
        row_sum: float = 0
        for j in range(dim1):
            row_sum += abs(A[i, j])
        row_sums[i] = row_sum

    return float(max(row_sums))


def vec_norm_inf(A: np.ndarray) -> float:
    return max(abs(A))


# Евклидова норма / норма Фробениуса
def norm_fro(A: np.ndarray) -> float | None:
    dim1, dim2 = A.shape
    # Проверка на квадратную матрицу
    if dim1 != dim2:
        return None

    el_sum_squared: float = 0
    for i in range(dim1):
        for j in range(dim1):
            el_sum_squared += pow(A[i, j], 2)

    return float(pow(el_sum_squared, 0.5))


def vec_norm_fro(A: np.ndarray) -> float:
    return float((pow(sum(pow(A, 2)), 0.5)))
