import numpy as np
import pytest

from methods.progonka_func import clean_solve_progonka_func


def make_grid(a: float, b: float, h: float) -> np.ndarray:
    n = int(round((b - a) / h))
    return np.array([a + i * h for i in range(n + 1)], dtype=float)


def assert_discrete_scheme_residuals_are_small(
    y: np.ndarray,
    h: float,
    p,
    q,
    f,
    a: float,
    b: float,
    alpha: np.ndarray,
    beta: np.ndarray,
    tol: float = 1e-9,
) -> None:
    x = make_grid(a, b, h)
    n = len(x) - 1

    alpha0, alpha1, A_value = alpha
    beta0, beta1, B_value = beta

    left_residual = alpha0 * y[0] + alpha1 * (y[1] - y[0]) / h - A_value
    right_residual = beta0 * y[n] + beta1 * (y[n] - y[n - 1]) / h - B_value

    assert left_residual == pytest.approx(0.0, abs=tol)
    assert right_residual == pytest.approx(0.0, abs=tol)

    for i in range(1, n):
        second_diff = (y[i + 1] - 2 * y[i] + y[i - 1]) / h**2
        first_diff = (y[i + 1] - y[i - 1]) / (2 * h)

        residual = second_diff + p(x[i]) * first_diff + q(x[i]) * y[i] - f(x[i])

        assert residual == pytest.approx(0.0, abs=tol)


def test_zero_step_returns_none():
    result = clean_solve_progonka_func(
        h=0.0,
        p=lambda x: 0.0,
        q=lambda x: 0.0,
        f=lambda x: 0.0,
        a=0.0,
        b=1.0,
        alpha=np.array([1.0, 0.0, 0.0]),
        beta=np.array([1.0, 0.0, 1.0]),
    )

    assert result is None


def test_too_coarse_grid_returns_none():
    result = clean_solve_progonka_func(
        h=1.0,
        p=lambda x: 0.0,
        q=lambda x: 0.0,
        f=lambda x: 0.0,
        a=0.0,
        b=1.0,
        alpha=np.array([1.0, 0.0, 0.0]),
        beta=np.array([1.0, 0.0, 1.0]),
    )

    assert result is None


def test_linear_solution_with_dirichlet_boundaries_is_exact():
    # y = 2x - 1, y'' = 0.
    h = 0.1
    a = -1.0
    b = 1.0
    x = make_grid(a, b, h)

    result = clean_solve_progonka_func(
        h=h,
        p=lambda x: 0.0,
        q=lambda x: 0.0,
        f=lambda x: 0.0,
        a=a,
        b=b,
        alpha=np.array([1.0, 0.0, 2 * a - 1]),
        beta=np.array([1.0, 0.0, 2 * b - 1]),
    )

    assert result is not None
    np.testing.assert_allclose(result, 2 * x - 1, atol=1e-10)


def test_linear_solution_with_robin_boundaries_is_exact():
    # y = 3x + 2, y'' = 0. First-order boundary differences are exact for a line.
    h = 0.1
    a = 0.0
    b = 1.0
    x = make_grid(a, b, h)

    result = clean_solve_progonka_func(
        h=h,
        p=lambda x: 0.0,
        q=lambda x: 0.0,
        f=lambda x: 0.0,
        a=a,
        b=b,
        alpha=np.array([0.0, 1.0, 3.0]),
        beta=np.array([2.0, 3.0, 19.0]),
    )

    assert result is not None
    np.testing.assert_allclose(result, 3 * x + 2, atol=1e-10)


def test_quadratic_solution_with_discrete_robin_boundary_is_exact():
    # y = x^2, y'' = 2. The right boundary value uses the same backward
    # difference that clean_solve_progonka_func uses internally.
    h = 0.1
    a = 0.0
    b = 1.0
    x = make_grid(a, b, h)
    y_exact = x**2

    right_discrete_value = 2 * y_exact[-1] + 3 * (y_exact[-1] - y_exact[-2]) / h

    result = clean_solve_progonka_func(
        h=h,
        p=lambda x: 0.0,
        q=lambda x: 0.0,
        f=lambda x: 2.0,
        a=a,
        b=b,
        alpha=np.array([1.0, 0.0, y_exact[0]]),
        beta=np.array([2.0, 3.0, right_discrete_value]),
    )

    assert result is not None
    np.testing.assert_allclose(result, y_exact, atol=1e-10)


def test_task_equation_matches_expected_values():
    result = clean_solve_progonka_func(
        h=0.1,
        p=lambda x: 1 / x,
        q=lambda x: 2.0,
        f=lambda x: x,
        a=0.7,
        b=1.0,
        alpha=np.array([1.0, 0.0, 0.5]),
        beta=np.array([2.0, 3.0, 1.2]),
    )

    expected = np.array(
        [
            0.5,
            0.5104985147322894,
            0.5176819380422897,
            0.5228268169146465,
        ]
    )

    assert result is not None
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_task_equation_satisfies_discrete_system():
    h = 0.1
    a = 0.7
    b = 1.0
    alpha = np.array([1.0, 0.0, 0.5])
    beta = np.array([2.0, 3.0, 1.2])

    p = lambda x: 1 / x
    q = lambda x: 2.0
    f = lambda x: x

    result = clean_solve_progonka_func(
        h=h,
        p=p,
        q=q,
        f=f,
        a=a,
        b=b,
        alpha=alpha,
        beta=beta,
    )

    assert result is not None
    assert_discrete_scheme_residuals_are_small(result, h, p, q, f, a, b, alpha, beta)


def test_input_boundary_arrays_are_not_mutated():
    alpha = np.array([1.0, 0.0, 0.5])
    beta = np.array([2.0, 3.0, 1.2])
    alpha_before = alpha.copy()
    beta_before = beta.copy()

    clean_solve_progonka_func(
        h=0.1,
        p=lambda x: 1 / x,
        q=lambda x: 2.0,
        f=lambda x: x,
        a=0.7,
        b=1.0,
        alpha=alpha,
        beta=beta,
    )

    np.testing.assert_array_equal(alpha, alpha_before)
    np.testing.assert_array_equal(beta, beta_before)
