import math

import numpy as np
import pandas as pd
import pytest

# ЗАМЕНИ на свой реальный путь
from methods.simpson_integration_func import (
    solve_simpson_integration_func,
)

from methods.first_diff_table import clean_solve_first_diff_table
from methods.second_diff_table import clean_solve_second_diff_table


def make_table(x: list[float], y: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["x", *x],
            ["y", *y],
        ]
    )


# ============================================================
# solve_simpson_integration_func
# ============================================================


def test_simpson_integrates_constant_function() -> None:
    result = solve_simpson_integration_func(
        f=lambda x: 5.0,
        a=0.0,
        b=10.0,
        eps=1e-8,
    )

    assert result is not None
    assert result == pytest.approx(50.0, rel=1e-10, abs=1e-10)


def test_simpson_integrates_linear_function() -> None:
    # ∫_0^4 x dx = 8
    result = solve_simpson_integration_func(
        f=lambda x: x,
        a=0.0,
        b=4.0,
        eps=1e-8,
    )

    assert result is not None
    assert result == pytest.approx(8.0, rel=1e-10, abs=1e-10)


def test_simpson_integrates_quadratic_function() -> None:
    # ∫_0^1 x^2 dx = 1/3
    result = solve_simpson_integration_func(
        f=lambda x: x**2,
        a=0.0,
        b=1.0,
        eps=1e-10,
    )

    assert result is not None
    assert result == pytest.approx(1.0 / 3.0, rel=1e-8, abs=1e-8)


def test_simpson_integrates_sin_function() -> None:
    # ∫_0^pi sin(x) dx = 2
    result = solve_simpson_integration_func(
        f=math.sin,
        a=0.0,
        b=math.pi,
        eps=1e-8,
    )

    assert result is not None
    assert result == pytest.approx(2.0, rel=1e-7, abs=1e-7)


def test_simpson_returns_none_for_bad_eps() -> None:
    result = solve_simpson_integration_func(
        f=lambda x: x**2,
        a=0.0,
        b=1.0,
        eps=0.0,
    )

    assert result is None


def test_simpson_returns_none_if_maxiter_too_small() -> None:
    result = solve_simpson_integration_func(
        f=math.sin,
        a=0.0,
        b=math.pi,
        eps=1e-15,
        maxiter=0,
    )

    assert result is None


# ============================================================
# clean_solve_first_diff_table
# ============================================================

# ============================================================
# clean_solve_second_diff_table
# ============================================================


def test_second_diff_for_quadratic_function_uniform_grid() -> None:
    # y = x^2
    # Вторая производная равна 2.
    # По центральной конечной разности:
    # (y[i + 2] - 2y[i + 1] + y[i]) / h^2 = 2
    x = [0.0, 1.0, 2.0, 3.0]
    y = [0.0, 1.0, 4.0, 9.0]

    table = make_table(x, y)
    x_dots = np.array([1.0, 2.0])

    result = clean_solve_second_diff_table(table, x_dots)

    assert result is not None
    np.testing.assert_allclose(
        result,
        np.array([2.0, 2.0]),
        rtol=1e-10,
        atol=1e-10,
    )


def test_second_diff_for_linear_function_uniform_grid() -> None:
    # y = 5x - 1
    # Вторая производная равна 0.
    x = [0.0, 1.0, 2.0, 3.0]
    y = [-1.0, 4.0, 9.0, 14.0]

    table = make_table(x, y)
    x_dots = np.array([1.0, 2.0])

    result = clean_solve_second_diff_table(table, x_dots)

    assert result is not None
    np.testing.assert_allclose(
        result,
        np.array([0.0, 0.0]),
        rtol=1e-10,
        atol=1e-10,
    )


def test_second_diff_returns_none_for_duplicate_x() -> None:
    x = [0.0, 1.0, 1.0, 2.0]
    y = [0.0, 1.0, 2.0, 4.0]

    table = make_table(x, y)
    x_dots = np.array([1.0])

    result = clean_solve_second_diff_table(table, x_dots)

    assert result is None

