from typing import Callable

import numpy as np
import pandas as pd
from .methods_utils import norm


def clean_solve_mpi_matrix(
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

        x_diff_norm: float = vec_norm(x_vec_new - x_vec_old)
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


def pretty_solve_mpi_matrix(
    eps: float, A: np.ndarray, b: np.ndarray, maxiter: int = 10000
) -> dict:

    # Чтобы не портить исходные данные
    A_matrix: np.ndarray = A.copy().astype(float)
    b_vec: np.ndarray = b.copy().astype(float)

    dim: int = A_matrix.shape[0]

    x_vec_old: np.ndarray = np.zeros(dim, dtype=float)
    x_vec_new: np.ndarray = np.zeros(dim, dtype=float)
    residual_vec: np.ndarray = np.zeros(dim, dtype=float)

    # Матрица перехода
    B_matrix: np.ndarray = A_matrix.copy()
    for i in range(dim):
        if abs(A_matrix[i, i]) < 1e-12:
            return {
                "status": "error",
                "message": "Диагональный элемент исходной матрицы не может быть нулем",
            }
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
    vec_norm: Callable[..., float] = norm.vec_norm_1
    correct_norm: float | None = None
    chosen_norm_name: str | None = None

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
            "c_vec": b_vec.copy(),
            "norms": norms,
        }

    # Начальное приближение
    x_vec_old = b_vec.copy()
    rows: list = []

    # Итерации МПИ
    iterations = 0
    while True:
        x_vec_new = b_vec.copy()
        for i in range(dim):
            for j in range(dim):
                if i != j:
                    x_vec_new[i] += B_matrix[i, j] * x_vec_old[j]
        iterations += 1

        x_diff_norm: float = vec_norm(x_vec_new - x_vec_old)

        rows.append(
            {
                "iteration": iterations,
                "||x_n - x_n-1||": x_diff_norm,
                "accuracy": correct_norm / (1 - correct_norm) * x_diff_norm,
            }
        )

        if correct_norm / (1 - correct_norm) * x_diff_norm < eps:
            break

        if iterations >= maxiter:
            return {
                "status": "error",
                "message": "Превышено максимальное число итераций",
                "transition_matrix": B_matrix.copy(),
                "transition_vector": b_vec.copy(),
                "transition_matrix_norms": norms.copy(),
                "chosen_norm": correct_norm,
                "iteration_table": pd.DataFrame(rows),
                "iterations": iterations,
                "chosen_norm_name": chosen_norm_name,
            }

        x_vec_old = x_vec_new.copy()

    # Невязки
    for i in range(dim):
        summa_for_residual: float = 0
        for j in range(dim):
            summa_for_residual += A[i, j] * x_vec_new[j]
        residual_vec[i] = b[i] - summa_for_residual

    return {
        "status": "ok",
        "solutions": x_vec_new.copy(),
        "residuals": residual_vec.copy(),
        "residuals_norm_inf": norm.vec_norm_inf(residual_vec.copy()),
        "transition_matrix": B_matrix.copy(),
        "transition_vector": b_vec.copy(),
        "transition_matrix_norms": norms.copy(),
        "chosen_norm": correct_norm,
        "iteration_table": pd.DataFrame(rows),
        "iterations": iterations,
        "chosen_norm_name": chosen_norm_name,
    }
