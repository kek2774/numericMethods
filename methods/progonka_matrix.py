import numpy as np
import pandas as pd


def clean_solve_progonka_matrix(
    A: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    tmp_eps = 1e-9
    dim = A.shape[0]

    # Чтобы не портить исходные данные
    A_matrix: np.ndarray = A.copy().astype(float)
    d_coeff: np.ndarray = b.copy().astype(float)

    x_vec: np.ndarray = np.zeros(dim, dtype=float)
    residual_vec: np.ndarray = np.zeros(dim, dtype=float)

    # Заполняем массивы a, b, c
    a_coeff: np.ndarray = np.zeros(dim, dtype=float)
    b_coeff: np.ndarray = np.zeros(dim, dtype=float)
    c_coeff: np.ndarray = np.zeros(dim, dtype=float)
    for i in range(dim):
        b_coeff[i] = A_matrix[i, i]

        if i > 0:
            a_coeff[i] = A_matrix[i, i - 1]

        if i < dim - 1:
            c_coeff[i] = A_matrix[i, i + 1]

    # Вычисление коэффициентов A, B

    if abs(b_coeff[0]) < tmp_eps:
        return None

    A_coeff: np.ndarray = np.zeros(dim, dtype=float)
    B_coeff: np.ndarray = np.zeros(dim, dtype=float)
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


def pretty_solve_progonka_matrix(A: np.ndarray, b: np.ndarray) -> dict:
    tmp_eps = 1e-9
    dim = A.shape[0]

    # Чтобы не портить исходные данные
    A_matrix: np.ndarray = A.copy().astype(float)
    d_coeff: np.ndarray = b.copy().astype(float)

    x_vec: np.ndarray = np.zeros(dim, dtype=float)
    residual_vec: np.ndarray = np.zeros(dim, dtype=float)

    dominance_rows: list[dict] = []
    coeff_rows: list[dict] = []
    forward_rows: list[dict] = []
    backward_rows: list[dict] = []
    residual_rows: list[dict] = []

    # Заполняем массивы a, b, c
    a_coeff: np.ndarray = np.zeros(dim, dtype=float)
    b_coeff: np.ndarray = np.zeros(dim, dtype=float)
    c_coeff: np.ndarray = np.zeros(dim, dtype=float)
    for i in range(dim):
        b_coeff[i] = A_matrix[i, i]

        if i > 0:
            a_coeff[i] = A_matrix[i, i - 1]

        if i < dim - 1:
            c_coeff[i] = A_matrix[i, i + 1]

        dominance_rows.append(
            {
                "i": i,
                "a_i": a_coeff[i],
                "b_i": b_coeff[i],
                "c_i": c_coeff[i],
                "|b_i|": abs(b_coeff[i]),
                "|a_i| + |c_i|": abs(a_coeff[i]) + abs(c_coeff[i]),
                "condition": abs(b_coeff[i]) >= abs(a_coeff[i]) + abs(c_coeff[i]),
            }
        )

    for i in range(dim):
        coeff_rows.append(
            {
                "i": i,
                "a_i": a_coeff[i],
                "b_i": b_coeff[i],
                "c_i": c_coeff[i],
                "d_i": d_coeff[i],
            }
        )

    # Вычисление коэффициентов A, B

    if abs(b_coeff[0]) < tmp_eps:
        return {
            "status": "error",
            "message": "Первый диагональный коэффициент слишком близок к нулю",
            "coefficient_table": pd.DataFrame(coeff_rows),
            "dominance_table": pd.DataFrame(dominance_rows),
        }

    A_coeff: np.ndarray = np.zeros(dim, dtype=float)
    B_coeff: np.ndarray = np.zeros(dim, dtype=float)
    A_coeff[0] = -c_coeff[0] / b_coeff[0]
    B_coeff[0] = d_coeff[0] / b_coeff[0]

    forward_rows.append(
        {
            "i": 0,
            "denominator": b_coeff[0],
            "A_i": A_coeff[0],
            "B_i": B_coeff[0],
        }
    )

    for i in range(1, dim):
        A_coeff[i] = -c_coeff[i] / (b_coeff[i] + a_coeff[i] * A_coeff[i - 1])
        B_coeff[i] = (d_coeff[i] - a_coeff[i] * B_coeff[i - 1]) / (
            b_coeff[i] + a_coeff[i] * A_coeff[i - 1]
        )

        forward_rows.append(
            {
                "i": i,
                "denominator": b_coeff[i] + a_coeff[i] * A_coeff[i - 1],
                "a_i": a_coeff[i],
                "b_i": b_coeff[i],
                "c_i": c_coeff[i],
                "d_i": d_coeff[i],
                "A_i": A_coeff[i],
                "B_i": B_coeff[i],
            }
        )

    # Вычисление решений
    x_vec[-1] = B_coeff[-1]
    backward_rows.append(
        {
            "i": dim - 1,
            "x_i": x_vec[-1],
            "formula": "x_n = B_n",
        }
    )

    for i in range(dim - 2, -1, -1):
        x_vec[i] = A_coeff[i] * x_vec[i + 1] + B_coeff[i]
        backward_rows.append(
            {
                "i": i,
                "A_i": A_coeff[i],
                "x_i+1": x_vec[i + 1],
                "B_i": B_coeff[i],
                "x_i": x_vec[i],
            }
        )

    # Вычисление невязок
    for i in range(dim):
        summa = 0
        for j in range(dim):
            summa += A_matrix[i, j] * x_vec[j]
        residual_vec[i] = b[i] - summa

        residual_rows.append(
            {
                "i": i,
                "Ax_i": summa,
                "b_i": b[i],
                "r_i = b_i - Ax_i": residual_vec[i],
                "|r_i|": abs(residual_vec[i]),
            }
        )

    return {
        "status": "ok",
        "solution": x_vec.copy(),
        "residuals": residual_vec.copy(),
        "residuals_norm_inf": float(np.max(np.abs(residual_vec))),
        "a_coeff": a_coeff.copy(),
        "b_coeff": b_coeff.copy(),
        "c_coeff": c_coeff.copy(),
        "d_coeff": d_coeff.copy(),
        "A_coeff": A_coeff.copy(),
        "B_coeff": B_coeff.copy(),
        "coefficient_table": pd.DataFrame(coeff_rows),
        "dominance_table": pd.DataFrame(dominance_rows),
        "forward_pass_table": pd.DataFrame(forward_rows),
        "backward_pass_table": pd.DataFrame(backward_rows),
        "residual_table": pd.DataFrame(residual_rows),
    }
