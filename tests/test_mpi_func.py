# tests/test_mpi_func.py

import math
import pytest

from methods.mpi_func import clean_solve_mpi_matrix


def call_no_crash(eps, f, a, b, maxiter=10000):
    try:
        return clean_solve_mpi_matrix(eps, f, a, b, maxiter=maxiter)
    except (ArithmeticError, ValueError, OverflowError, ZeroDivisionError) as exc:
        pytest.fail(f"solve_mpi raised {type(exc).__name__}: {exc}")


def assert_root(result, f, expected, x_tol=1e-6, f_tol=1e-6):
    assert result is not None, "expected root, got None"
    assert math.isfinite(result), "root must be finite"
    assert abs(result - expected) <= x_tol
    assert abs(f(result)) <= f_tol


@pytest.mark.parametrize("eps", [0.0, -1e-9, -100.0])
def test_invalid_eps_returns_none(eps):
    f = lambda x: x * x - 2

    assert clean_solve_mpi_matrix(eps, f, 1.0, 2.0) is None


@pytest.mark.parametrize("maxiter", [0, -1, -100])
def test_invalid_maxiter_returns_none(maxiter):
    f = lambda x: x * x - 2

    result = call_no_crash(
        eps=1e-8,
        f=f,
        a=1.0,
        b=2.0,
        maxiter=maxiter,
    )

    assert result is None


@pytest.mark.parametrize(
    "root",
    [-10.0, -1.5, -1e-6, 0.0, 0.25, 2.0, 100.0],
)
def test_linear_roots(root):
    f = lambda x, r=root: x - r

    result = call_no_crash(
        eps=1e-10,
        f=f,
        a=root - 2.0,
        b=root + 3.0,
        maxiter=100,
    )

    assert_root(result, f, root, x_tol=1e-7, f_tol=1e-7)


@pytest.mark.parametrize(
    "f, a, b, expected",
    [
        (
            lambda x: x * x - 2,
            1.0,
            2.0,
            math.sqrt(2),
        ),
        (
            lambda x: x**3 - x - 2,
            1.0,
            2.0,
            1.5213797068045676,
        ),
        (
            lambda x: math.exp(x) - 3,
            0.0,
            2.0,
            math.log(3),
        ),
        (
            lambda x: math.log(x) - 1,
            2.0,
            4.0,
            math.e,
        ),
    ],
)
def test_standard_monotonic_cases(f, a, b, expected):
    result = call_no_crash(
        eps=1e-8,
        f=f,
        a=a,
        b=b,
        maxiter=10000,
    )

    assert_root(result, f, expected, x_tol=1e-5, f_tol=1e-5)


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
            lambda x: x - 2,
            -10.0,
            2.0,
            2.0,
        ),
        (
            lambda x: x * x - 4,
            2.0,
            5.0,
            2.0,
        ),
    ],
)
def test_root_on_interval_boundary(f, a, b, expected):
    result = call_no_crash(
        eps=1e-8,
        f=f,
        a=a,
        b=b,
        maxiter=1000,
    )

    assert_root(result, f, expected, x_tol=1e-7, f_tol=1e-7)


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
            lambda x: math.exp(x) + 1,
            -2.0,
            2.0,
        ),
    ],
)
def test_no_root_or_no_sign_change_returns_none(f, a, b):
    result = call_no_crash(
        eps=1e-8,
        f=f,
        a=a,
        b=b,
        maxiter=100,
    )

    assert result is None


def test_derivative_changes_sign_returns_none():
    # f'(x) = 2x меняет знак на [-0.5, 2.0].
    # На границах корня нет, но внутри есть корень x = 1.
    # Если solve_mpi проверяет условие сжатия через get_phi_pars,
    # он должен отказаться от такого интервала.
    f = lambda x: x * x - 1

    result = call_no_crash(
        eps=1e-8,
        f=f,
        a=-0.5,
        b=2.0,
        maxiter=1000,
    )

    assert result is None


def test_zero_derivative_inside_interval_can_converge():
    # f'(x) = 3x^2, в x = 0 производная нулевая,
    # но корень x = 1 находится нормально.
    f = lambda x: x**3 - 1

    result = call_no_crash(
        eps=1e-8,
        f=f,
        a=-0.5,
        b=2.0,
        maxiter=1000,
    )

    assert_root(result, f, 1.0, x_tol=1e-5, f_tol=1e-5)


def test_very_small_eps():
    f = lambda x: x * x - 2

    result = call_no_crash(
        eps=1e-10,
        f=f,
        a=1.0,
        b=2.0,
        maxiter=10000,
    )

    assert_root(result, f, math.sqrt(2), x_tol=1e-6, f_tol=1e-6)


def test_low_maxiter_returns_none():
    f = lambda x: x * x - 2

    result = call_no_crash(
        eps=1e-12,
        f=f,
        a=1.0,
        b=2.0,
        maxiter=1,
    )

    assert result is None


def test_result_stays_finite_for_hard_function():
    f = lambda x: math.exp(x) - 10

    result = call_no_crash(
        eps=1e-8,
        f=f,
        a=1.0,
        b=3.0,
        maxiter=10000,
    )

    assert_root(result, f, math.log(10), x_tol=1e-5, f_tol=1e-5)


def test_root_very_close_to_boundary():
    expected = 1e-6
    f = lambda x: x - expected

    result = call_no_crash(
        eps=1e-10,
        f=f,
        a=0.0,
        b=1.0,
        maxiter=1000,
    )

    assert_root(result, f, expected, x_tol=1e-7, f_tol=1e-7)
