import numpy as np
import pandas as pd
from typing import Callable

from methods.methods_utils import matrix_diag
from .newton_table import P_n, get_divided_difference_matrix
from .runge_cutta_func import get_k_coeffs


def clean_solve_adams_func(
    f: Callable[..., float], h: float, x_0: float, y_0: float
) -> Callable[[float], float] | None:

    if h == 0:
        return None

    x_plus: list[float] = [x_0]
    y_plus: list[float] = [y_0]

    x_minus: list[float] = [x_0]
    y_minus: list[float] = [y_0]

    def interpolated_func(x_star: float) -> float:

        if x_star >= x_0:
            x: list[float] = x_plus
            y: list[float] = y_plus
            h_local: float = h
        else:
            x = x_minus
            y = y_minus
            h_local = -h

        if len(x) == 1:
            k_s: np.ndarray = get_k_coeffs(f, x_0, y_0, h_local)
            x.append(x_0 + h_local)
            y.append(y_0 + h_local / 6 * (k_s[0] + 2 * k_s[1] + 2 * k_s[2] + k_s[3]))

        while abs(x[-1] - x_0) < abs(x_star - x_0) + 3 * abs(h):
            x_old = x[-1]
            y_old = y[-1]
            y.append(
                y_old + h_local * (3 / 2 * f(x_old, y_old) - 1 / 2 * f(x[-2], y[-2]))
            )
            x.append(x_old + h_local)

        index: int = int(abs((x_star - x_0) / h))

        left: int = index - 1
        right: int = index + 3

        if left < 0:
            left = 0
            right = 4

        x_local: np.ndarray = np.array(x[left:right])
        y_local: np.ndarray = np.array(y[left:right])

        div_diffs: np.ndarray = get_divided_difference_matrix(x_local, y_local)
        a_coeffs: np.ndarray = matrix_diag.get_diag_as_vector(div_diffs[1:, :])

        return P_n(x_star, a_coeffs, x_local, a_coeffs.size)

    return interpolated_func


def pretty_solve_adams_func(
    f: Callable[..., float], h: float, x_0: float, y_0: float, x_star: float
) -> dict:

    if h == 0:
        return {
            "status": "error",
            "message": "Шаг h не должен быть равен нулю",
        }

    try:
        x_star = float(x_star)
    except Exception:
        return {
            "status": "error",
            "message": "Не удалось прочитать x_star",
        }

    x: list[float] = [x_0]
    y: list[float] = [y_0]
    h_local: float = h if x_star >= x_0 else -h
    rows: list[dict] = []

    k_s: np.ndarray = get_k_coeffs(f, x_0, y_0, h_local)
    x.append(x_0 + h_local)
    y.append(y_0 + h_local / 6 * (k_s[0] + 2 * k_s[1] + 2 * k_s[2] + k_s[3]))

    rows.append(
        {
            "i": 0,
            "method": "runge-cutta",
            "x_i": x_0,
            "y_i": y_0,
            "k1": k_s[0],
            "k2": k_s[1],
            "k3": k_s[2],
            "k4": k_s[3],
            "x_i+1": x[1],
            "y_i+1": y[1],
        }
    )

    while abs(x[-1] - x_0) < abs(x_star - x_0) + 3 * abs(h):
        x_old: float = x[-1]
        y_old: float = y[-1]
        f_i: float = f(x_old, y_old)
        f_i_prev: float = f(x[-2], y[-2])
        y_new: float = y_old + h_local * (3 / 2 * f_i - 1 / 2 * f_i_prev)
        x_new: float = x_old + h_local

        rows.append(
            {
                "i": len(x) - 1,
                "method": "adams",
                "x_i-1": x[-2],
                "y_i-1": y[-2],
                "f_i-1": f_i_prev,
                "x_i": x_old,
                "y_i": y_old,
                "f_i": f_i,
                "x_i+1": x_new,
                "y_i+1": y_new,
            }
        )

        y.append(y_new)
        x.append(x_new)

    index: int = int(abs((x_star - x_0) / h))

    left: int = index - 1
    right: int = index + 3

    if left < 0:
        left = 0
        right = 4

    x_local: np.ndarray = np.array(x[left:right])
    y_local: np.ndarray = np.array(y[left:right])

    div_diffs: np.ndarray = get_divided_difference_matrix(x_local, y_local)
    a_coeffs: np.ndarray = matrix_diag.get_diag_as_vector(div_diffs[1:, :])
    result: float = P_n(x_star, a_coeffs, x_local, a_coeffs.size)

    interpolation_rows: list[dict] = []
    for i in range(len(x_local)):
        interpolation_rows.append(
            {
                "i": i,
                "x_i": x_local[i],
                "y_i": y_local[i],
                "a_i": a_coeffs[i],
            }
        )

    return {
        "status": "ok",
        "x_star": x_star,
        "result": result,
        "h": h,
        "x_0": x_0,
        "y_0": y_0,
        "x": np.array(x, dtype=float).copy(),
        "y": np.array(y, dtype=float).copy(),
        "x_local": x_local.copy(),
        "y_local": y_local.copy(),
        "coefficients": a_coeffs.copy(),
        "divided_differences_matrix": div_diffs.copy(),
        "step_table": pd.DataFrame(rows),
        "interpolation_table": pd.DataFrame(interpolation_rows),
    }
