import numpy as np
import pandas as pd
from typing import Callable

from .progonka_matrix import clean_solve_progonka_matrix, pretty_solve_progonka_matrix
from .methods_utils import matrix_diag


def clean_solve_progonka_func(
    h: float,
    p: Callable[..., float],
    q: Callable[..., float],
    f: Callable[..., float],
    a: float,
    b: float,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray | None:
    if h == 0:
        return None

    n: int = int(round((b - a) / h))

    if n < 2:
        return None

    dim: int = n + 1

    x: np.ndarray = np.array([a + i * h for i in range(dim)], dtype=float)

    A_matrix: np.ndarray = np.zeros((dim, dim), dtype=float)
    b_vec: np.ndarray = np.zeros(dim, dtype=float)

    alpha0: float = alpha[0]
    alpha1: float = alpha[1]
    A_value: float = alpha[2]

    beta0: float = beta[0]
    beta1: float = beta[1]
    B_value: float = beta[2]

    # Левое краевое условие
    A_matrix[0] = matrix_diag.build_diag_row(
        0,
        alpha0 - alpha1 / h,
        alpha1 / h,
        dim,
        0,
    )
    b_vec[0] = A_value

    # Внутренние узлы
    for i in range(1, n):
        p_i: float = p(x[i])
        q_i: float = q(x[i])
        f_i: float = f(x[i])

        left: float = 1 - p_i * h / 2
        center: float = -2 + q_i * h**2
        right: float = 1 + p_i * h / 2

        A_matrix[i] = matrix_diag.build_diag_row(left, center, right, dim, i)
        b_vec[i] = f_i * h**2

    # Правое краевое условие
    A_matrix[n] = matrix_diag.build_diag_row(
        -beta1 / h,
        beta0 + beta1 / h,
        0,
        dim,
        n,
    )
    b_vec[n] = B_value

    result: tuple | None = clean_solve_progonka_matrix(A_matrix, b_vec)

    if result is None:
        return None

    return result[0]


def pretty_solve_progonka_func(
    h: float,
    p: Callable[..., float],
    q: Callable[..., float],
    f: Callable[..., float],
    a: float,
    b: float,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> dict:
    if h == 0:
        return {
            "status": "error",
            "message": "Шаг h не должен быть равен нулю",
        }

    n: int = int(round((b - a) / h))

    if n < 2:
        return {
            "status": "error",
            "message": "Для краевой задачи нужно минимум 3 узла сетки",
        }

    if len(alpha) < 3 or len(beta) < 3:
        return {
            "status": "error",
            "message": "Краевые условия должны задаваться массивами [coef0, coef1, value]",
        }

    dim: int = n + 1
    x: np.ndarray = np.array([a + i * h for i in range(dim)], dtype=float)

    A_matrix: np.ndarray = np.zeros((dim, dim), dtype=float)
    b_vec: np.ndarray = np.zeros(dim, dtype=float)

    alpha0: float = alpha[0]
    alpha1: float = alpha[1]
    A_value: float = alpha[2]

    beta0: float = beta[0]
    beta1: float = beta[1]
    B_value: float = beta[2]

    if abs(alpha0) + abs(alpha1) == 0:
        return {
            "status": "error",
            "message": "Левое краевое условие задано некорректно",
        }

    if abs(beta0) + abs(beta1) == 0:
        return {
            "status": "error",
            "message": "Правое краевое условие задано некорректно",
        }

    grid_rows: list[dict] = []
    equation_rows: list[dict] = []
    boundary_rows: list[dict] = []

    for i in range(dim):
        grid_rows.append(
            {
                "i": i,
                "x_i": x[i],
            }
        )

    A_matrix[0] = matrix_diag.build_diag_row(
        0,
        alpha0 - alpha1 / h,
        alpha1 / h,
        dim,
        0,
    )
    b_vec[0] = A_value

    boundary_rows.append(
        {
            "side": "left",
            "alpha0": alpha0,
            "alpha1": alpha1,
            "value": A_value,
            "A[0,0]": A_matrix[0, 0],
            "A[0,1]": A_matrix[0, 1] if dim > 1 else 0,
            "b[0]": b_vec[0],
        }
    )

    for i in range(1, n):
        p_i: float = p(x[i])
        q_i: float = q(x[i])
        f_i: float = f(x[i])

        left: float = 1 - p_i * h / 2
        center: float = -2 + q_i * h**2
        right: float = 1 + p_i * h / 2

        A_matrix[i] = matrix_diag.build_diag_row(left, center, right, dim, i)
        b_vec[i] = f_i * h**2

        equation_rows.append(
            {
                "i": i,
                "x_i": x[i],
                "p_i": p_i,
                "q_i": q_i,
                "f_i": f_i,
                "left": left,
                "center": center,
                "right": right,
                "right_part": b_vec[i],
            }
        )

    A_matrix[n] = matrix_diag.build_diag_row(
        -beta1 / h,
        beta0 + beta1 / h,
        0,
        dim,
        n,
    )
    b_vec[n] = B_value

    boundary_rows.append(
        {
            "side": "right",
            "beta0": beta0,
            "beta1": beta1,
            "value": B_value,
            "A[n,n-1]": A_matrix[n, n - 1],
            "A[n,n]": A_matrix[n, n],
            "b[n]": b_vec[n],
        }
    )

    progonka_result: dict = pretty_solve_progonka_matrix(A_matrix, b_vec)

    if progonka_result["status"] != "ok":
        progonka_result.update(
            {
                "x": x.copy(),
                "A_matrix": A_matrix.copy(),
                "b_vec": b_vec.copy(),
                "grid_table": pd.DataFrame(grid_rows),
                "equation_table": pd.DataFrame(equation_rows),
                "boundary_table": pd.DataFrame(boundary_rows),
            }
        )

        return progonka_result

    solution: np.ndarray = progonka_result["solution"].copy()

    return {
        "status": "ok",
        "x": x.copy(),
        "y": solution.copy(),
        "h": h,
        "a": a,
        "b": b,
        "alpha": alpha.copy(),
        "beta": beta.copy(),
        "A_matrix": A_matrix.copy(),
        "b_vec": b_vec.copy(),
        "residuals": progonka_result["residuals"].copy(),
        "residuals_norm_inf": progonka_result["residuals_norm_inf"],
        "grid_table": pd.DataFrame(grid_rows),
        "equation_table": pd.DataFrame(equation_rows),
        "boundary_table": pd.DataFrame(boundary_rows),
        "coefficient_table": progonka_result["coefficient_table"],
        "dominance_table": progonka_result["dominance_table"],
        "forward_pass_table": progonka_result["forward_pass_table"],
        "backward_pass_table": progonka_result["backward_pass_table"],
        "residual_table": progonka_result["residual_table"],
    }
