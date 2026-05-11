import numpy as np
import pandas as pd
import pytest

from methods.newton_table import clean_solve_newton_table


def make_table(xs, ys) -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["x", *xs],
            ["y", *ys],
        ]
    )


def lagrange_oracle(xs, ys, x_star: float) -> float:
    result = 0.0
    n = len(xs)

    for i in range(n):
        term = float(ys[i])

        for j in range(n):
            if i != j:
                term *= (x_star - xs[j]) / (xs[i] - xs[j])

        result += term

    return result


def poly_value(coeffs, x):
    return sum(c * x**i for i, c in enumerate(coeffs))


@pytest.mark.parametrize(
    "xs, ys, x_star, expected",
    [
        ([0, 1], [2, 5], 0.0, 2.0),
        ([0, 1], [2, 5], 1.0, 5.0),
        ([0, 1], [2, 5], 0.5, 3.5),
        ([-2, 2], [-1, 7], 0.0, 3.0),
    ],
)
def test_linear_interpolation(xs, ys, x_star, expected):
    table = make_table(xs, ys)

    actual = clean_solve_newton_table(table, x_star)

    assert actual == pytest.approx(expected, abs=1e-10)


def test_quadratic_polynomial_exact():
    xs = np.array([-2.0, 0.0, 1.0])
    ys = np.array([2 * x**2 - 3 * x + 1 for x in xs])
    x_star = 0.5

    table = make_table(xs, ys)

    actual = clean_solve_newton_table(table, x_star)
    expected = 2 * x_star**2 - 3 * x_star + 1

    assert actual == pytest.approx(expected, abs=1e-10)


def test_cubic_polynomial_exact_non_uniform_nodes():
    xs = np.array([-3.0, -0.5, 1.25, 4.0])
    ys = np.array([-(x**3) + 2 * x**2 - 5 * x + 7 for x in xs])
    x_star = 2.2

    table = make_table(xs, ys)

    actual = clean_solve_newton_table(table, x_star)
    expected = -(x_star**3) + 2 * x_star**2 - 5 * x_star + 7

    assert actual == pytest.approx(expected, abs=1e-8)


def test_high_degree_polynomial_exact_many_nodes():
    coeffs = [4.0, -2.0, 0.5, 0.0, 3.0, -0.25]

    xs = np.array([-2.0, -1.0, 0.0, 0.75, 2.0, 3.0])
    ys = np.array([poly_value(coeffs, x) for x in xs])

    table = make_table(xs, ys)

    for x_star in [-1.5, 0.3, 1.4, 2.5]:
        actual = clean_solve_newton_table(table, x_star)
        expected = poly_value(coeffs, x_star)

        assert actual == pytest.approx(expected, abs=1e-7)


def test_exactly_at_table_nodes_returns_original_y_values():
    xs = np.array([-3.0, -1.0, 0.0, 2.0, 5.0])
    ys = np.array([9.0, 1.0, 0.0, 4.0, 25.0])

    table = make_table(xs, ys)

    for x_star, expected_y in zip(xs, ys):
        actual = clean_solve_newton_table(table, x_star)

        assert actual == pytest.approx(expected_y, abs=1e-10)


def test_result_matches_lagrange_oracle_for_arbitrary_data():
    xs = np.array([-2.0, -0.7, 0.1, 1.5, 3.0])
    ys = np.array([3.2, -1.1, 0.4, 2.8, -5.0])

    table = make_table(xs, ys)

    for x_star in [-1.5, 0.0, 0.8, 2.2]:
        actual = clean_solve_newton_table(table, x_star)
        expected = lagrange_oracle(xs, ys, x_star)

        assert actual == pytest.approx(expected, abs=1e-8)


def test_extrapolation_outside_interval():
    xs = np.array([-1.0, 0.0, 1.0, 2.0])
    ys = np.array([5.0, 1.0, -1.0, 1.0])

    table = make_table(xs, ys)

    for x_star in [-3.0, 4.0]:
        actual = clean_solve_newton_table(table, x_star)
        expected = lagrange_oracle(xs, ys, x_star)

        assert actual == pytest.approx(expected, abs=1e-8)


def test_unsorted_x_nodes_still_correct():
    xs = np.array([2.0, -1.0, 0.5, 4.0, 0.0])
    ys = np.array([x**3 - 2 * x + 1 for x in xs])

    table = make_table(xs, ys)
    x_star = 1.25

    actual = clean_solve_newton_table(table, x_star)
    expected = x_star**3 - 2 * x_star + 1

    assert actual == pytest.approx(expected, abs=1e-8)


def test_single_point_table_returns_constant_value():
    table = make_table([10.0], [42.0])

    actual = clean_solve_newton_table(table, -999.0)

    assert actual == pytest.approx(42.0, abs=1e-10)


def test_function_does_not_mutate_input_table():
    table = make_table([-1.0, 0.0, 1.0], [1.0, 0.0, 1.0])
    original = table.copy(deep=True)

    clean_solve_newton_table(table, 0.5)

    pd.testing.assert_frame_equal(table, original)
