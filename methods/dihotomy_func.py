from typing import Callable
import pandas as pd
from tenacity import P


def clean_solve_dihotomy_func(
    eps: float, f: Callable[..., float], a: float, b: float
) -> float | None:
    if eps <= 0:
        return None

    # Проверка существования решения на отрезке [a, b]
    f_a: float = f(a)
    f_b: float = f(b)
    if abs(f_a) < 1e-12:
        return a
    if abs(f_b) < 1e-12:
        return b
    if f_a * f_b > 0:
        return None

    # Итерации метода дихотомии
    while abs(a - b) >= 2 * eps:
        x_0: float = (a + b) / 2
        f_x_0: float = f(x_0)
        if f_x_0 * f(a) < 0:
            b = x_0
        elif f_x_0 * f(b) < 0:
            a = x_0
        elif abs(f_x_0) < 1e-12:
            return x_0

    x: float = (a + b) / 2
    return x


def pretty_solve_dihotomy_func(
    eps: float, f: Callable[..., float], a: float, b: float
) -> dict:
    if eps <= 0:
        return {
            "status": "error",
            "message": "Точность должна быть положительной",
        }

    rows: list = []

    # Проверка существования решения на отрезке [a, b]
    f_a: float = f(a)
    f_b: float = f(b)
    if abs(f_a) < 1e-12:
        return {
            "status": "ok",
            "solution": a,
        }
    if abs(f_b) < 1e-12:
        return {
            "status": "ok",
            "solution": b,
        }
    if f_a * f_b > 0:
        return {
            "status": "error",
            "message": "На введенном отрезке не существует решения",
        }

    # Итерации метода дихотомии
    while abs(a - b) >= 2 * eps:
        x_0: float = (a + b) / 2
        f_x_0: float = f(x_0)

        rows.append(
            {
                "x_0": x_0,
                "f(x_0)": f_x_0,
                "a": a,
                "f(a)": f(a),
                "b": b,
                "f(b)": f(b),
                "|a - b|": abs(a - b),
            }
        )

        if f_x_0 * f(a) < 0:
            b = x_0
        elif f_x_0 * f(b) < 0:
            a = x_0
        elif abs(f_x_0) < 1e-12:
            return {
                "status": "ok",
                "solution": x_0,
                "iteration_table": pd.DataFrame(rows),
            }

    return {
        "status": "ok",
        "solution": (a + b) / 2,
        "iteration_table": pd.DataFrame(rows),
    }
