import math

import pytest

from methods.dihotomy_func import clean_solve_dihotomy_func


def assert_root_close(result, f, expected_root, eps):
    assert result is not None, "Ожидался корень, но функция вернула None"
    assert isinstance(result, float), "Функция должна вернуть float или None"

    assert abs(result - expected_root) <= 2 * eps, (
        f"\nОжидался корень около: {expected_root}"
        f"\nПолучено: {result}"
        f"\nПогрешность: {abs(result - expected_root)}"
    )

    assert abs(f(result)) <= 1e-5, (
        f"\nЗначение функции в найденной точке должно быть близко к 0"
        f"\nf({result}) = {f(result)}"
    )


def test_linear_function():
    def f(x):
        return x - 2.0

    eps = 1e-8

    result = clean_solve_dihotomy_func(eps, f, 0.0, 5.0)

    assert_root_close(result, f, expected_root=2.0, eps=eps)


def test_quadratic_function_positive_root():
    def f(x):
        return x**2 - 4.0

    eps = 1e-8

    result = clean_solve_dihotomy_func(eps, f, 0.0, 5.0)

    assert_root_close(result, f, expected_root=2.0, eps=eps)


def test_quadratic_function_negative_root():
    def f(x):
        return x**2 - 4.0

    eps = 1e-8

    result = clean_solve_dihotomy_func(eps, f, -5.0, 0.0)

    assert_root_close(result, f, expected_root=-2.0, eps=eps)


def test_sin_function_root():
    def f(x):
        return math.sin(x)

    eps = 1e-8

    result = clean_solve_dihotomy_func(eps, f, 3.0, 4.0)

    assert_root_close(result, f, expected_root=math.pi, eps=eps)


def test_root_at_left_border():
    def f(x):
        return x - 1.0

    result = clean_solve_dihotomy_func(1e-8, f, 1.0, 5.0)

    assert result == 1.0


def test_root_at_right_border():
    def f(x):
        return x - 5.0

    result = clean_solve_dihotomy_func(1e-8, f, 1.0, 5.0)

    assert result == 5.0


def test_no_root_on_interval_returns_none():
    def f(x):
        return x**2 + 1.0

    result = clean_solve_dihotomy_func(1e-8, f, -2.0, 2.0)

    assert result is None


def test_no_sign_change_returns_none():
    def f(x):
        return x**2 - 4.0

    result = clean_solve_dihotomy_func(1e-8, f, 3.0, 5.0)

    assert result is None


@pytest.mark.parametrize(
    "f,a,b,expected_root",
    [
        (lambda x: x - 10.0, 0.0, 20.0, 10.0),
        (lambda x: 2.0 * x + 6.0, -10.0, 10.0, -3.0),
        (lambda x: x**3 - 8.0, 0.0, 5.0, 2.0),
        (lambda x: math.exp(x) - 2.0, 0.0, 2.0, math.log(2.0)),
    ],
)
def test_parametrized_roots(f, a, b, expected_root):
    eps = 1e-8

    result = clean_solve_dihotomy_func(eps, f, a, b)

    assert_root_close(result, f, expected_root, eps)


def test_accuracy_depends_on_eps():
    def f(x):
        return x - math.sqrt(2.0)

    result_rough = clean_solve_dihotomy_func(1e-3, f, 0.0, 2.0)
    result_precise = clean_solve_dihotomy_func(1e-8, f, 0.0, 2.0)

    assert result_rough is not None
    assert result_precise is not None

    exact = math.sqrt(2.0)

    assert abs(result_precise - exact) < abs(result_rough - exact)
