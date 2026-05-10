from typing import Callable
from .methods_utils import diffs
import numpy as np
import pandas as pd


def phi(f: Callable[..., float], x: float, lambda_value: float) -> float:
    return x + lambda_value * f(x)


def get_phi_pars(
    f: Callable[..., float], a: float, b: float, points_count: int = 1000
) -> None | tuple[float, float]:
    points = np.linspace(a, b, points_count)
    ders: np.ndarray = np.array(
        [diffs.derivative_n(f, x, 1) for x in points], dtype=float
    )
    df_min = float(np.min(ders))
    df_max = float(np.max(ders))

    if df_min <= 0 <= df_max:
        return None

    lambda_value: float = -1.8 / (df_max + df_min)

    q_vals: np.ndarray = abs(1 + lambda_value * ders)
    q = float(max(q_vals))  # | 1 + lambda * f'(x) |

    if q >= 1:
        return None

    return (lambda_value, q)


def clean_solve_mpi_matrix(
    eps: float, f: Callable[..., float], a: float, b: float, maxiter: int = 10000
) -> float | None:

    if eps <= 0:
        return None

    if f(a) * f(b) > 0:
        return None
    if abs(f(a)) < 1e-12:
        return a
    if abs(f(b)) < 1e-12:
        return b

    phi_pars: None | tuple[float, float] = get_phi_pars(f, a, b)
    if phi_pars is None:
        return None

    lambda_val, q = phi_pars

    x_0: float = (a + b) / 2
    x_old: float = x_0
    x_new: float = phi(f, x_old, lambda_val)
    iterations: int = 1

    while q / (1 - q) * abs(x_new - x_old) >= eps:
        x_old = x_new
        x_new = phi(f, x_old, lambda_val)
        iterations += 1

        if iterations >= maxiter:
            return None

    return x_new


def pretty_solve_mpi_matrix(
    eps: float, f: Callable[..., float], a: float, b: float, maxiter: int = 10000
) -> dict:

    if eps <= 0:
        return {
            "status": "error",
            "message": "Точность должна быть положительной",
        }

    if f(a) * f(b) > 0:
        return {
            "status": "error",
            "message": "На введенном отрезке не существует решения",
        }

    if abs(f(a)) < 1e-12:
        return {
            "status": "ok",
            "solution": a,
        }
    if abs(f(b)) < 1e-12:
        return {
            "status": "ok",
            "solution": b,
        }

    phi_pars: None | tuple[float, float] = get_phi_pars(f, a, b)
    if phi_pars is None:
        return {
            "status": "error",
            "message": "Не удалось определить параметры для функции phi",
        }

    lambda_val, q = phi_pars

    rows: list = []
    x_0: float = (a + b) / 2
    x_old: float = x_0
    x_new: float = phi(f, x_old, lambda_val)
    iterations: int = 1

    rows.append(
        {
            "iteration": iterations,
            "x_n": x_0,
            "x_n+1": x_new,
            "|x_n+1 - x_n|": abs(x_new - x_old),
            "accuracy": q / (1 - q) * abs(x_new - x_old),
        }
    )

    while q / (1 - q) * abs(x_new - x_old) >= eps:
        x_old = x_new
        x_new = phi(f, x_old, lambda_val)
        iterations += 1

        if iterations >= maxiter:
            return {
                "status": "error",
                "message": "Превышено максимальное число итераций",
            }

        rows.append(
            {
                "iteration": iterations,
                "x_n": x_old,
                "x_n+1": x_new,
                "|x_n+1 - x_n|": abs(x_new - x_old),
                "accuracy": q / (1 - q) * abs(x_new - x_old),
            }
        )

    return {
        "status": "ok",
        "solution": x_new,
        "iteration_table": pd.DataFrame(rows),
        "q": q,
        "lambda_val": lambda_val,
    }
