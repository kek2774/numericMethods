from typing import Callable
from .methods_utils import diffs
import pandas as pd


def clean_solve_newton_func(
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

    x_0: float = 0

    if f(a) * diffs.derivative_n(f, a, 2) > 0:
        x_0 = a
    elif f(b) * diffs.derivative_n(f, b, 2) > 0:
        x_0 = b
    else:
        x_0 = (a + b) / 2

    x_old: float = x_0
    if abs(f(x_old)) < 1e-12:
        return x_old

    x_new: float = x_old - f(x_old) / diffs.derivative_n(f, x_old, 1)
    iterations: int = 1
    while abs(x_new - x_old) >= eps:
        x_old = x_new

        if abs(f(x_old)) < 1e-12:
            return x_old

        derr: float = diffs.derivative_n(f, x_old, 1)
        x_new = x_old - f(x_old) / derr

        iterations += 1

        if iterations >= maxiter:
            return None

    return x_new


def pretty_solve_newton_func(
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
            "message": "На концах отрезка нет смены знака, наличие корня не подтверждено",
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

    rows: list = []

    x_0: float = 0

    if f(a) * diffs.derivative_n(f, a, 2) > 0:
        x_0 = a
    elif f(b) * diffs.derivative_n(f, b, 2) > 0:
        x_0 = b
    else:
        x_0 = (a + b) / 2

    x_old: float = x_0
    if abs(f(x_old)) < 1e-12:
        return {"status": "ok", "solution": x_old}

    x_new: float = x_old - f(x_old) / diffs.derivative_n(f, x_old, 1)
    iterations: int = 1

    rows.append(
        {
            "iterations": iterations,
            "x_n": x_0,
            "f(x_n)": f(x_old),
            "f'(x_n)": diffs.derivative_n(f, x_old, 1),
            "x_n+1": x_new,
            "|x_n+1 - x_n|": abs(x_new - x_old),
        }
    )

    while abs(x_new - x_old) >= eps:
        x_old = x_new

        if abs(f(x_old)) < 1e-12:
            return {
                "status": "ok",
                "solution": x_old,
                "iteration_table": pd.DataFrame(rows),
                "iterations": iterations,
            }

        derr: float = diffs.derivative_n(f, x_old, 1)

        if abs(derr) < 1e-12:
            return {
                "status": "error",
                "message": "Производная близка к нулю, шаг Ньютона невозможен",
                "iteration_table": pd.DataFrame(rows),
            }

        x_new = x_old - f(x_old) / derr

        iterations += 1
        rows.append(
            {
                "iterations": iterations,
                "x_n": x_old,
                "f(x_n)": f(x_old),
                "f'(x_n)": derr,
                "x_n+1": x_new,
                "|x_n+1 - x_n|": abs(x_new - x_old),
            }
        )

        if iterations >= maxiter:
            return {
                "status": "error",
                "message": "Превышено максимально число итераций",
                "iteration_table": pd.DataFrame(rows),
                "iterations": iterations,
            }

    return {
        "status": "ok",
        "solution": x_new,
        "f''(a)": diffs.derivative_n(f, a, 2),
        "f''(b)": diffs.derivative_n(f, b, 2),
        "x_0": x_0,
        "iterations": iterations,
        "iteration_table": pd.DataFrame(rows),
    }
