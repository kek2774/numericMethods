from typing import Callable


def solve_dihotomy_func(
    eps: float, f: Callable[..., float], a: float, b: float
) -> float | None:
    if abs(eps) <= 0:
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
