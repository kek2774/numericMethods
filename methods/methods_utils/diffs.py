from typing import Callable


def derivative_n(f: Callable[..., float], x: float, n: int, h: float = 1e-5) -> float:
    if n == 0:
        return f(x)

    return (derivative_n(f, x + h, n - 1, h) - derivative_n(f, x - h, n - 1, h)) / (
        2 * h
    )
