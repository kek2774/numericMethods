import numpy as np
import pandas as pd
import pytest

# поменяй импорт под свой проект
from methods.cube_spline_table import clean_solve_cubic_splines_table


def make_table(x: list[float], y: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["x", *x],
            ["y", *y],
        ]
    )


def normalize_coeffs(result: list | None, intervals_count: int) -> np.ndarray:
    """
    Твоя функция возвращает:

    [
        a_array,
        b_array,
        c_array,
        d_array,
    ]

    Этот helper превращает это в таблицу:

    [
        [a0, b0, c0, d0],
        [a1, b1, c1, d1],
        ...
    ]

    То есть в форму (n - 1, 4).
    """

    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 4

    a = np.asarray(result[0], dtype=float)
    b = np.asarray(result[1], dtype=float)
    c = np.asarray(result[2], dtype=float)
    d = np.asarray(result[3], dtype=float)

    # У тебя a и c могут быть длины n, а b и d длины n - 1.
    # Для проверки сплайна нужны коэффициенты только по интервалам.
    assert len(a) >= intervals_count
    assert len(b) >= intervals_count
    assert len(c) >= intervals_count
    assert len(d) >= intervals_count

    a = a[:intervals_count]
    b = b[:intervals_count]
    c = c[:intervals_count]
    d = d[:intervals_count]

    return np.column_stack((a, b, c, d))


def spline_value(coeffs: np.ndarray, i: int, t: float) -> float:
    a, b, c, d = coeffs[i]
    return a + b * t + c * t**2 + d * t**3


def spline_first_derivative(coeffs: np.ndarray, i: int, t: float) -> float:
    _, b, c, d = coeffs[i]
    return b + 2 * c * t + 3 * d * t**2


def spline_second_derivative(coeffs: np.ndarray, i: int, t: float) -> float:
    _, _, c, d = coeffs[i]
    return 2 * c + 6 * d * t


def test_duplicate_x_returns_none():
    table = make_table(
        x=[0.0, 1.0, 1.0, 2.0],
        y=[0.0, 1.0, 2.0, 3.0],
    )

    result = clean_solve_cubic_splines_table(table)

    assert result is None


def test_too_few_points_returns_none():
    table = make_table(
        x=[1.0],
        y=[5.0],
    )

    result = clean_solve_cubic_splines_table(table)

    assert result is None


def test_two_points_gives_straight_line():
    x = [0.0, 4.0]
    y = [2.0, 10.0]

    table = make_table(x, y)
    result = clean_solve_cubic_splines_table(table)

    coeffs = normalize_coeffs(result, intervals_count=1)

    expected = np.array(
        [
            [2.0, 2.0, 0.0, 0.0],
        ]
    )

    np.testing.assert_allclose(coeffs, expected, atol=1e-10)


def test_constant_function_gives_constant_spline():
    x = [-3.0, -1.0, 0.0, 2.0, 5.0]
    y = [7.0, 7.0, 7.0, 7.0, 7.0]

    table = make_table(x, y)
    result = clean_solve_cubic_splines_table(table)

    coeffs = normalize_coeffs(result, intervals_count=len(x) - 1)

    expected = np.array(
        [
            [7.0, 0.0, 0.0, 0.0],
            [7.0, 0.0, 0.0, 0.0],
            [7.0, 0.0, 0.0, 0.0],
            [7.0, 0.0, 0.0, 0.0],
        ]
    )

    np.testing.assert_allclose(coeffs, expected, atol=1e-10)


def test_linear_function_gives_exact_linear_spline_on_nonuniform_grid():
    x = [-5.0, -2.0, -0.5, 3.0, 10.0]
    y = [2.5 * xi - 3.0 for xi in x]

    table = make_table(x, y)
    result = clean_solve_cubic_splines_table(table)

    coeffs = normalize_coeffs(result, intervals_count=len(x) - 1)

    for i in range(len(x) - 1):
        expected_a = y[i]
        expected_b = 2.5
        expected_c = 0.0
        expected_d = 0.0

        np.testing.assert_allclose(
            coeffs[i],
            [expected_a, expected_b, expected_c, expected_d],
            atol=1e-10,
        )

    for i in range(len(x) - 1):
        h = x[i + 1] - x[i]
        t = h / 2

        expected_value = 2.5 * (x[i] + t) - 3.0
        actual_value = spline_value(coeffs, i, t)

        assert actual_value == pytest.approx(expected_value, abs=1e-10)


def test_known_three_point_natural_spline_coefficients():
    """
    Узлы:
    x = [0, 1, 2]
    y = [0, 1, 0]

    Натуральный кубический сплайн:

    S0(t) = 0 + 1.5t + 0t^2 - 0.5t^3
    S1(t) = 1 + 0t - 1.5t^2 + 0.5t^3
    """

    x = [0.0, 1.0, 2.0]
    y = [0.0, 1.0, 0.0]

    table = make_table(x, y)
    result = clean_solve_cubic_splines_table(table)

    coeffs = normalize_coeffs(result, intervals_count=2)

    expected = np.array(
        [
            [0.0, 1.5, 0.0, -0.5],
            [1.0, 0.0, -1.5, 0.5],
        ]
    )

    np.testing.assert_allclose(coeffs, expected, atol=1e-10)


def test_spline_passes_through_all_nodes():
    x = [-2.0, -0.7, 0.0, 0.9, 2.4, 5.0]
    y = [3.0, -1.0, 0.5, 2.2, -0.3, 4.1]

    table = make_table(x, y)
    result = clean_solve_cubic_splines_table(table)

    coeffs = normalize_coeffs(result, intervals_count=len(x) - 1)

    for i in range(len(x) - 1):
        h = x[i + 1] - x[i]

        left_value = spline_value(coeffs, i, 0.0)
        right_value = spline_value(coeffs, i, h)

        assert left_value == pytest.approx(y[i], abs=1e-9)
        assert right_value == pytest.approx(y[i + 1], abs=1e-9)


def test_first_derivative_is_continuous_at_inner_nodes():
    x = [-2.0, -0.7, 0.0, 0.9, 2.4, 5.0]
    y = [3.0, -1.0, 0.5, 2.2, -0.3, 4.1]

    table = make_table(x, y)
    result = clean_solve_cubic_splines_table(table)

    coeffs = normalize_coeffs(result, intervals_count=len(x) - 1)

    for i in range(len(x) - 2):
        h = x[i + 1] - x[i]

        left_derivative = spline_first_derivative(coeffs, i, h)
        right_derivative = spline_first_derivative(coeffs, i + 1, 0.0)

        assert left_derivative == pytest.approx(right_derivative, abs=1e-8)


def test_second_derivative_is_continuous_at_inner_nodes():
    x = [-2.0, -0.7, 0.0, 0.9, 2.4, 5.0]
    y = [3.0, -1.0, 0.5, 2.2, -0.3, 4.1]

    table = make_table(x, y)
    result = clean_solve_cubic_splines_table(table)

    coeffs = normalize_coeffs(result, intervals_count=len(x) - 1)

    for i in range(len(x) - 2):
        h = x[i + 1] - x[i]

        left_second = spline_second_derivative(coeffs, i, h)
        right_second = spline_second_derivative(coeffs, i + 1, 0.0)

        assert left_second == pytest.approx(right_second, abs=1e-8)


def test_natural_boundary_conditions_are_zero_second_derivatives():
    x = [-2.0, -0.7, 0.0, 0.9, 2.4, 5.0]
    y = [3.0, -1.0, 0.5, 2.2, -0.3, 4.1]

    table = make_table(x, y)
    result = clean_solve_cubic_splines_table(table)

    coeffs = normalize_coeffs(result, intervals_count=len(x) - 1)

    first_left_second = spline_second_derivative(coeffs, 0, 0.0)

    last_i = len(x) - 2
    last_h = x[-1] - x[-2]
    last_right_second = spline_second_derivative(coeffs, last_i, last_h)

    assert first_left_second == pytest.approx(0.0, abs=1e-8)
    assert last_right_second == pytest.approx(0.0, abs=1e-8)


def test_spline_has_finite_coefficients_and_values_inside_intervals():
    x = [-10.0, -3.0, -1.0, 0.2, 4.5, 20.0]
    y = [100.0, -5.0, 3.0, 2.0, 40.0, -10.0]

    table = make_table(x, y)
    result = clean_solve_cubic_splines_table(table)

    coeffs = normalize_coeffs(result, intervals_count=len(x) - 1)

    assert np.all(np.isfinite(coeffs))

    for i in range(len(x) - 1):
        h = x[i + 1] - x[i]

        for t in [0.0, h * 0.25, h * 0.5, h * 0.75, h]:
            value = spline_value(coeffs, i, t)
            first = spline_first_derivative(coeffs, i, t)
            second = spline_second_derivative(coeffs, i, t)

            assert np.isfinite(value)
            assert np.isfinite(first)
            assert np.isfinite(second)


def test_raw_result_has_four_coefficient_vectors():
    x = [0.0, 1.0, 2.0, 3.0]
    y = [0.0, 2.0, 1.0, 3.0]

    table = make_table(x, y)
    result = clean_solve_cubic_splines_table(table)

    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 4

    for coeff_vector in result:
        arr = np.asarray(coeff_vector, dtype=float)

        assert arr.ndim == 1
        assert np.all(np.isfinite(arr))


def test_input_table_is_not_mutated():
    x = [0.0, 1.0, 2.0, 3.0]
    y = [0.0, 2.0, 1.0, 3.0]

    table = make_table(x, y)
    table_before = table.copy(deep=True)

    clean_solve_cubic_splines_table(table)

    pd.testing.assert_frame_equal(table, table_before)
