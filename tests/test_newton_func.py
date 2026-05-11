# test_newton_hardcore.py

import math
import pytest

# замени main на имя своего файла
from methods.newton_func import clean_solve_newton_func


def call_no_crash(eps, f, a, b, maxiter=10000):
    try:
        return clean_solve_newton_func(eps, f, a, b, maxiter=maxiter)
    except (ArithmeticError, ValueError, OverflowError) as exc:
        pytest.fail(f"solve_newton_func raised {type(exc).__name__}: {exc}")


def assert_root(result, f, expected, x_tol=1e-6, f_tol=1e-6):
    assert result is not None, "expected root, got None"
    assert math.isfinite(result), "root must be finite"
    assert abs(result - expected) <= x_tol
    assert abs(f(result)) <= f_tol


@pytest.mark.parametrize("eps", [0.0, -1e-9, -100.0])
def test_invalid_eps_returns_none(eps):
    f = lambda x: x * x - 2
    assert clean_solve_newton_func(eps, f, 1, 2) is None


@pytest.mark.parametrize(
    "root",
    [-10.0, -1.5, -1e-6, 0.0, 0.25, 2.0, 100.0],
)
def test_linear_roots_converge_fast(root):
    f = lambda x, r=root: x - r

    result = call_no_crash(
        eps=1e-12,
        f=f,
        a=root - 3.7,
        b=root + 9.1,
        maxiter=10,
    )

    assert_root(result, f, root, x_tol=1e-10, f_tol=1e-10)


@pytest.mark.parametrize(
    "name, f, a, b, expected",
    [
        (
            "sqrt_2",
            lambda x: x * x - 2,
            1.0,
            2.0,
            math.sqrt(2),
        ),
        (
            "cubic",
            lambda x: x**3 - x - 2,
            1.0,
            2.0,
            1.5213797068045676,
        ),
        (
            "cos_x_minus_x",
            lambda x: math.cos(x) - x,
            0.0,
            1.0,
            0.7390851332151607,
        ),
        (
            "sin_root_pi",
            lambda x: math.sin(x),
            3.0,
            4.0,
            math.pi,
        ),
        (
            "exp_root_ln3",
            lambda x: math.exp(x) - 3,
            0.0,
            2.0,
            math.log(3),
        ),
        (
            "log_root_e",
            lambda x: math.log(x) - 1,
            2.0,
            4.0,
            math.e,
        ),
    ],
)
def test_standard_convergence_cases(name, f, a, b, expected):
    result = call_no_crash(
        eps=1e-10,
        f=f,
        a=a,
        b=b,
        maxiter=100,
    )

    assert_root(result, f, expected, x_tol=1e-6, f_tol=1e-6)


@pytest.mark.parametrize(
    "f, a, b, expected",
    [
        (
            lambda x: x - 2,
            2.0,
            10.0,
            2.0,
        ),
        (
            lambda x: x**3,
            0.0,
            1.0,
            0.0,
        ),
        (
            lambda x: (x - 4) * (x + 1),
            -2.0,
            4.0,
            4.0,
        ),
    ],
)
def test_root_on_interval_boundary(f, a, b, expected):
    result = call_no_crash(
        eps=1e-10,
        f=f,
        a=a,
        b=b,
        maxiter=100,
    )

    assert_root(result, f, expected, x_tol=1e-8, f_tol=1e-8)


@pytest.mark.parametrize(
    "f, a, b",
    [
        (
            lambda x: x * x + 1,
            -5.0,
            5.0,
        ),
        (
            lambda x: x - 10,
            0.0,
            1.0,
        ),
        (
            lambda x: (x - 1) ** 2 + 0.001,
            0.0,
            2.0,
        ),
    ],
)
def test_no_root_on_interval_returns_none(f, a, b):
    result = call_no_crash(
        eps=1e-8,
        f=f,
        a=a,
        b=b,
        maxiter=50,
    )

    assert result is None


def test_zero_derivative_at_start_does_not_crash():
    # На [-0.5, 0.5] корня нет.
    # Если бездумно взять середину x=0, то f'(0)=0.
    f = lambda x: x**4 - 1

    result = call_no_crash(
        eps=1e-8,
        f=f,
        a=-0.5,
        b=0.5,
        maxiter=20,
    )

    assert result is None


def test_multiple_root_converges_slowly_but_correctly():
    # Кратный корень: метод Ньютона сходится медленнее.
    f = lambda x: (x - 1) ** 3

    result = call_no_crash(
        eps=1e-10,
        f=f,
        a=0.0,
        b=2.0,
        maxiter=1000,
    )

    assert_root(result, f, 1.0, x_tol=1e-4, f_tol=1e-10)


def test_very_small_eps():
    f = lambda x: x * x - 2

    result = call_no_crash(
        eps=1e-12,
        f=f,
        a=1.0,
        b=2.0,
        maxiter=100,
    )

    assert_root(result, f, math.sqrt(2), x_tol=1e-9, f_tol=1e-9)


def test_maxiter_limits_iterations():
    f = lambda x: math.cos(x) - x

    result = call_no_crash(
        eps=1e-15,
        f=f,
        a=0.0,
        b=1.0,
        maxiter=1,
    )

    assert result is None


def test_maxiter_zero_returns_none():
    f = lambda x: x * x - 2

    result = call_no_crash(
        eps=1e-8,
        f=f,
        a=1.0,
        b=2.0,
        maxiter=0,
    )

    assert result is None


def test_root_very_close_to_boundary():
    expected = 1e-9
    f = lambda x: x - expected

    result = call_no_crash(
        eps=1e-12,
        f=f,
        a=0.0,
        b=1.0,
        maxiter=20,
    )

    assert_root(result, f, expected, x_tol=1e-10, f_tol=1e-10)
