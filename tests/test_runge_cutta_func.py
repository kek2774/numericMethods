import math

import pytest

from methods.runge_cutta_func import clean_solve_runge_cutta_func


def reference_rk4(f, h: float, x_0: float, y_0: float, x_star: float) -> float:
    x = x_0
    y = y_0

    if x_star == x_0:
        return y

    step = h if x_star > x_0 else -h
    steps_count = int(abs((x_star - x_0) / h))

    for _ in range(steps_count):
        k1 = f(x, y)
        k2 = f(x + step / 2, y + step * k1 / 2)
        k3 = f(x + step / 2, y + step * k2 / 2)
        k4 = f(x + step, y + step * k3)

        y += step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        x += step

    rest = x_star - x

    if abs(rest) > 1e-15:
        k1 = f(x, y)
        k2 = f(x + rest / 2, y + rest * k1 / 2)
        k3 = f(x + rest / 2, y + rest * k2 / 2)
        k4 = f(x + rest, y + rest * k3)

        y += rest / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return y


def task_equation(x: float, y: float) -> float:
    return 1 + 0.2 * y * math.sin(x) - y**2


def test_returns_none_for_zero_step():
    result = clean_solve_runge_cutta_func(lambda x, y: x + y, 0.0, 0.0, 1.0)

    assert result is None


def test_constant_derivative_exact_in_both_directions():
    # y' = -2.5, y(3) = 7 => y = 14.5 - 2.5x
    f = lambda x, y: -2.5
    solution = clean_solve_runge_cutta_func(f, 0.4, 3.0, 7.0)

    assert solution is not None

    for x_star in [-10.2, -3.6, 0.0, 2.6, 3.0, 4.8, 12.4]:
        expected = 14.5 - 2.5 * x_star

        assert solution(x_star) == pytest.approx(expected, abs=1e-10)


def test_cubic_solution_is_exact_between_nodes_in_both_directions():
    # y' = 3x^2, y(0) = 0 => y = x^3.
    # RK4 gives exact node values, and Newton interpolation by 4 points restores cubic exactly.
    solution = clean_solve_runge_cutta_func(lambda x, y: 3 * x**2, 0.2, 0.0, 0.0)

    assert solution is not None

    for x_star in [-2.3, -1.0, -0.25, -0.05, 0.0, 0.15, 0.7, 2.3]:
        assert solution(x_star) == pytest.approx(x_star**3, abs=1e-10)


def test_exponential_equation_matches_exact_solution_forward_and_backward():
    # y' = y, y(0) = 1 => y = e^x.
    solution = clean_solve_runge_cutta_func(lambda x, y: y, 0.05, 0.0, 1.0)

    assert solution is not None

    for x_star in [-2.0, -1.0, -0.5, -0.05, 0.0, 0.05, 0.5, 1.0, 2.0]:
        assert solution(x_star) == pytest.approx(math.exp(x_star), abs=1e-6)


def test_repeated_calls_on_different_sides_do_not_break_cached_points():
    solution = clean_solve_runge_cutta_func(lambda x, y: y, 0.05, 0.0, 1.0)

    assert solution is not None

    x_values = [2.0, -2.0, 0.25, -0.25, 1.5, -1.5, 0.0, 1.0, -1.0]

    for x_star in x_values:
        assert solution(x_star) == pytest.approx(math.exp(x_star), abs=1e-6)


@pytest.mark.parametrize(
    "x_star",
    [
        -2.0,
        -1.0,
        -0.55,
        -0.1,
        0.0,
        0.1,
        0.55,
        1.0,
        2.0,
    ],
)
def test_task_equation_matches_small_step_reference(x_star):
    solution = clean_solve_runge_cutta_func(task_equation, 0.1, 0.0, 0.0)

    assert solution is not None

    actual = solution(x_star)
    expected = reference_rk4(task_equation, 0.0005, 0.0, 0.0, x_star)

    assert actual == pytest.approx(expected, abs=1e-5)


def test_far_points_do_not_use_global_high_degree_interpolation():
    solution = clean_solve_runge_cutta_func(task_equation, 0.1, 0.0, 0.0)

    assert solution is not None

    for x_star in [-10.0, 10.0]:
        actual = solution(x_star)
        expected = reference_rk4(task_equation, 0.0005, 0.0, 0.0, x_star)

        assert math.isfinite(actual)
        assert actual == pytest.approx(expected, abs=1e-5)
