from typing import Callable, Literal

import numpy as np
import pandas as pd


def get_int_by_n(n: int, f: Callable[..., float], a: float, b: float) -> float:

    h: float = (b - a) / n
    x: np.ndarray = np.zeros(n + 1, dtype=float)
    y: np.ndarray = np.zeros(n + 1, dtype=float)
    x[0] = a
    y[0] = f(x[0])
    x[-1] = b
    y[-1] = f(x[-1])

    for i in range(1, n):
        x[i] = x[0] + i * h
        y[i] = f(x[i])

    S: float = h / 3 * (y[0] + y[-1])
    for i in range(1, n):
        factor: Literal[2] | Literal[4] = 2 if i % 2 == 0 else 4
        S += y[i] * factor * (h / 3)

    return S


def solve_simpson_integration_func(
    f: Callable[..., float], a: float, b: float, eps: float, maxiter: int = 1000
) -> float | None:

    if eps <= 0:
        return None

    if maxiter <= 0:
        return None

    minus_flag_int: bool = False
    if a > b:
        a, b = b, a
        minus_flag_int = True

    if a == b:
        return 0

    n: int = 2
    S_curr: float = get_int_by_n(n, f, a, b)
    n *= 2
    S_next: float = get_int_by_n(n, f, a, b)

    iterations: int = 1
    while abs(S_next - S_curr) / 15 > eps:
        S_curr = S_next
        n *= 2
        S_next = get_int_by_n(n, f, a, b)
        iterations += 1

        if iterations >= maxiter:
            return None

    return S_next if not minus_flag_int else -1 * S_next


def pretty_solve_simpson_integration_func(
    f: Callable[..., float], a: float, b: float, eps: float, maxiter: int = 1000
) -> dict:

    if eps <= 0:
        return {
            "status": "error",
            "message": "Точность должна быть положительной",
        }

    if maxiter <= 0:
        return {
            "status": "error",
            "message": "Максимальное число итераций должно быть положительным",
        }

    minus_flag_int: bool = False
    a_original: float = a
    b_original: float = b

    if a > b:
        a, b = b, a
        minus_flag_int = True

    if a == b:
        return {
            "status": "ok",
            "result": 0.0,
            "a": a_original,
            "b": b_original,
            "iteration_table": pd.DataFrame([]),
        }

    rows: list[dict] = []

    n: int = 2
    S_curr: float = get_int_by_n(n, f, a, b)
    n *= 2
    S_next: float = get_int_by_n(n, f, a, b)

    iterations: int = 1

    rows.append(
        {
            "iteration": iterations,
            "n_prev": n // 2,
            "S_prev": S_curr,
            "n": n,
            "S": S_next,
            "|S - S_prev| / 15": abs(S_next - S_curr) / 15,
        }
    )

    while abs(S_next - S_curr) / 15 > eps:
        S_curr = S_next
        n *= 2
        S_next = get_int_by_n(n, f, a, b)
        iterations += 1

        rows.append(
            {
                "iteration": iterations,
                "n_prev": n // 2,
                "S_prev": S_curr,
                "n": n,
                "S": S_next,
                "|S - S_prev| / 15": abs(S_next - S_curr) / 15,
            }
        )

        if iterations >= maxiter:
            return {
                "status": "error",
                "message": "Достигнуто максимальное число итераций",
                "a": a_original,
                "b": b_original,
                "eps": eps,
                "iteration_table": pd.DataFrame(rows),
            }

    result: float = S_next if not minus_flag_int else -1 * S_next

    return {
        "status": "ok",
        "result": result,
        "a": a_original,
        "b": b_original,
        "eps": eps,
        "iterations": iterations,
        "n": n,
        "iteration_table": pd.DataFrame(rows),
    }
