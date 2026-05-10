from typing import Callable

import numpy as np
import pandas as pd
from .methods_utils import norm
from .methods_utils import matrix_diag


def clean_solve_zeidel_matrix(
    eps: float, A: np.ndarray, b: np.ndarray, maxiter: int = 10000
) -> tuple[np.ndarray, np.ndarray] | None:

    # Чтобы не портить исходные данные
    A_matrix: np.ndarray = A.copy().astype(float)
    b_vec: np.ndarray = b.copy().astype(float)

    dim: int = A_matrix.shape[0]

    x_vec_old: np.ndarray = np.zeros(dim, dtype=float)
    x_vec_new: np.ndarray = np.zeros(dim, dtype=float)
    residual_vec: np.ndarray = np.zeros(dim, dtype=float)

    # Матрица ниже диагонали
    B_matrix: np.ndarray = matrix_diag.get_down_diag_matrix(A_matrix)

    # Наддиагональная матрица
    C_matrix: np.ndarray = matrix_diag.get_upp_diag_matrix(A_matrix)

    # Матрица перехода
    try:
        B_inv: np.ndarray = np.linalg.inv(B_matrix)
    except np.linalg.LinAlgError:
        return None

    T_matrix: np.ndarray = -B_inv @ C_matrix
    d_vec: np.ndarray = B_inv @ b_vec

    # Считаем все нормы
    norms: np.ndarray = np.array(
        [
            norm.norm_1(T_matrix),
            norm.norm_inf(T_matrix),
            norm.norm_fro(T_matrix),
        ],
        dtype=float,
    )
    vec_norm: Callable[..., float] = norm.vec_norm_1
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
    x_vec_old = d_vec.copy()

    # Итерации метода Зейделя
    iterations = 0
    while True:
        x_vec_new = d_vec.copy()
        for i in range(dim):
            for j in range(dim):
                x_vec_new[i] += T_matrix[i, j] * x_vec_old[j]
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


def pretty_solve_zeidel_matrix(
    eps: float, A: np.ndarray, b: np.ndarray, maxiter: int = 10000
) -> dict:

    # Чтобы не портить исходные данные
    A_matrix: np.ndarray = A.copy().astype(float)
    b_vec: np.ndarray = b.copy().astype(float)

    dim: int = A_matrix.shape[0]

    x_vec_old: np.ndarray = np.zeros(dim, dtype=float)
    x_vec_new: np.ndarray = np.zeros(dim, dtype=float)
    residual_vec: np.ndarray = np.zeros(dim, dtype=float)

    iteration_rows: list = []
    residual_rows: list = []
    chosen_norm_name = None

    # Матрица ниже диагонали
    B_matrix: np.ndarray = matrix_diag.get_down_diag_matrix(A_matrix)

    # Наддиагональная матрица
    C_matrix: np.ndarray = matrix_diag.get_upp_diag_matrix(A_matrix)

    # Матрица перехода
    try:
        B_inv: np.ndarray = np.linalg.inv(B_matrix)
    except np.linalg.LinAlgError:
        return {"status": "error", "message": "Матрица перехода не обращаема"}

    T_matrix: np.ndarray = -B_inv @ C_matrix
    d_vec: np.ndarray = B_inv @ b_vec

    # Считаем все нормы
    norms: np.ndarray = np.array(
        [
            norm.norm_1(T_matrix),
            norm.norm_inf(T_matrix),
            norm.norm_fro(T_matrix),
        ],
        dtype=float,
    )
    vec_norm: Callable[..., float] = norm.vec_norm_1
    correct_norm = None

    # Проверка на достаточное условие сходимости
    for i in range(len(norms)):
        if norms[i] < 1 and i == 0:
            chosen_norm_name = "norm_1"
            correct_norm = norms[i]
            vec_norm = norm.vec_norm_1
        if norms[i] < 1 and i == 1:
            chosen_norm_name = "norm_inf"
            correct_norm = norms[i]
            vec_norm = norm.vec_norm_inf
        if norms[i] < 1 and i == 2:
            chosen_norm_name = "norm_fro"
            correct_norm = norms[i]
            vec_norm = norm.vec_norm_fro

    if correct_norm is None:
        return {
            "status": "error",
            "message": "Достаточное условие сходимости не выполнено",
            "B_matrix": B_matrix.copy(),
            "C_matrix": C_matrix.copy(),
            "T_matrix": T_matrix.copy(),
            "d_vec": d_vec.copy(),
            "transition_matrix_norms": norms.copy(),
        }

    # Начальное приближение
    x_vec_old = d_vec.copy()

    # Итерации метода Зейделя
    iterations = 0
    while True:
        x_vec_new = d_vec.copy()
        for i in range(dim):
            for j in range(dim):
                x_vec_new[i] += T_matrix[i, j] * x_vec_old[j]
        iterations += 1

        x_diff_norm = vec_norm(x_vec_new - x_vec_old)
        row_data: dict = {
            "iteration": iterations,
            "x_diff_norm": x_diff_norm,
            "error_estimate": correct_norm / (1 - correct_norm) * x_diff_norm,
        }

        if correct_norm / (1 - correct_norm) * x_diff_norm < eps:
            break

        if iterations >= maxiter:
            return {
                "status": "error",
                "message": "Превышено максимальное число итераций",
                "B_matrix": B_matrix.copy(),
                "C_matrix": C_matrix.copy(),
                "T_matrix": T_matrix.copy(),
                "d_vec": d_vec.copy(),
                "transition_matrix_norms": norms.copy(),
                "chosen_norm": correct_norm,
                "chosen_norm_name": chosen_norm_name,
                "iteration_table": pd.DataFrame(iteration_rows),
                "iterations": iterations,
            }
        iteration_rows.append(row_data)

        x_vec_old = x_vec_new.copy()

    # Невязки
    for i in range(dim):
        summa_for_residual: float = 0
        for j in range(dim):
            summa_for_residual += A[i, j] * x_vec_new[j]
        residual_vec[i] = b[i] - summa_for_residual

        residual_rows.append(
            {
                "i": i,
                "Ax_i": summa_for_residual,
                "b_i": b[i],
                "r_i = b_i - Ax_i": residual_vec[i],
                "|r_i|": abs(residual_vec[i]),
            }
        )

    return {
        "status": "ok",
        "solutions": x_vec_new.copy(),
        "residuals": residual_vec.copy(),
        "residuals_norm_inf": norm.vec_norm_inf(residual_vec.copy()),
        "B_matrix": B_matrix.copy(),
        "C_matrix": C_matrix.copy(),
        "T_matrix": T_matrix.copy(),
        "d_vec": d_vec.copy(),
        "transition_matrix_norms": norms.copy(),
        "chosen_norm": correct_norm,
        "chosen_norm_name": chosen_norm_name,
        "iteration_table": pd.DataFrame(iteration_rows),
        "residual_table": pd.DataFrame(residual_rows),
        "iterations": iterations,
    }
