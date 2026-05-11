import numpy as np
import pandas as pd
import pytest

from methods.mls_quadratic_table import clean_solve_quadratic_mls_table


def make_table(x, y) -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["x", *x],
            ["y", *y],
        ]
    )


def expected_quadratic_mls(x, y) -> tuple[float, float, float]:
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)

    matrix = np.column_stack([x_arr**2, x_arr, np.ones(len(x_arr))])
    a, b, c = np.linalg.lstsq(matrix, y_arr, rcond=None)[0]

    return float(a), float(b), float(c)


@pytest.mark.parametrize(
    "x, y",
    [
        (
            [-3, -2, -1, 0, 1, 2, 3],
            [14, 7, 2, -1, -2, -1, 2],
        ),
        (
            [-5, -3, -1, 1, 3, 5],
            [25, 9, 1, 1, 9, 25],
        ),
        (
            [-10, -5, 0, 5, 10],
            [4, 4, 4, 4, 4],
        ),
        (
            [-4, -2, 0, 2, 4, 6],
            [-13, -7, -1, 5, 11, 17],
        ),
        (
            [0.1, 0.2, 0.4, 0.8, 1.6],
            [1.03, 1.12, 1.35, 2.01, 4.08],
        ),
        (
            [5, -2, 10, 0, 3, -7],
            [60.2, 3.1, 151.4, 1.2, 22.3, 83.8],
        ),
        (
            [1, 1, 2, 2, 3, 3, 4, 4],
            [2.1, 1.9, 5.2, 4.8, 10.1, 9.9, 17.2, 16.8],
        ),
        (
            [-1000, -500, 0, 500, 1000],
            [2999001, 749001, 1, 751001, 3001001],
        ),
        (
            [-2.5, -1.25, 0.0, 1.25, 2.5],
            [7.5, 2.1875, 1.0, 3.9375, 11.0],
        ),
        (
            [10, 20, 30, 40, 50, 60],
            [305.5, 1008.1, 2110.7, 3614.2, 5516.8, 7819.5],
        ),
    ],
)
def test_solve_quadratic_mls_table_matches_lstsq_reference(
    x: list[float], y: list[float]
) -> None:
    table = make_table(x, y)

    result = clean_solve_quadratic_mls_table(table)

    assert result is not None

    actual_a, actual_b, actual_c = result
    expected_a, expected_b, expected_c = expected_quadratic_mls(x, y)

    assert actual_a == pytest.approx(expected_a, rel=1e-10, abs=1e-9)
    assert actual_b == pytest.approx(expected_b, rel=1e-10, abs=1e-9)
    assert actual_c == pytest.approx(expected_c, rel=1e-10, abs=1e-9)


def test_solve_quadratic_mls_table_exact_parabola() -> None:
    x = [-3, -2, -1, 0, 1, 2, 3]
    y = [14, 7, 2, -1, -2, -1, 2]

    table = make_table(x, y)

    result = clean_solve_quadratic_mls_table(table)

    assert result is not None

    a, b, c = result

    assert a == pytest.approx(1.0)
    assert b == pytest.approx(-2.0)
    assert c == pytest.approx(-1.0)


def test_solve_quadratic_mls_table_linear_data() -> None:
    x = [-4, -2, 0, 2, 4, 6]
    y = [-13, -7, -1, 5, 11, 17]

    table = make_table(x, y)

    result = clean_solve_quadratic_mls_table(table)

    assert result is not None

    a, b, c = result

    assert a == pytest.approx(0.0, abs=1e-12)
    assert b == pytest.approx(3.0)
    assert c == pytest.approx(-1.0)


def test_solve_quadratic_mls_table_constant_data() -> None:
    x = [-100, -50, 0, 50, 100]
    y = [12.75, 12.75, 12.75, 12.75, 12.75]

    table = make_table(x, y)

    result = clean_solve_quadratic_mls_table(table)

    assert result is not None

    a, b, c = result

    assert a == pytest.approx(0.0, abs=1e-12)
    assert b == pytest.approx(0.0, abs=1e-12)
    assert c == pytest.approx(12.75)


def test_solve_quadratic_mls_table_unsorted_x_values() -> None:
    x = [8, -3, 5, 0, 12, -7, 2]
    y = [129.3, 19.2, 77.1, 1.4, 217.7, 92.9, 11.8]

    table = make_table(x, y)

    result = clean_solve_quadratic_mls_table(table)

    assert result is not None

    actual_a, actual_b, actual_c = result
    expected_a, expected_b, expected_c = expected_quadratic_mls(x, y)

    assert actual_a == pytest.approx(expected_a, rel=1e-10, abs=1e-10)
    assert actual_b == pytest.approx(expected_b, rel=1e-10, abs=1e-10)
    assert actual_c == pytest.approx(expected_c, rel=1e-10, abs=1e-10)


def test_solve_quadratic_mls_table_repeated_x_values_valid_case() -> None:
    x = [1, 1, 1, 2, 2, 3, 3, 4]
    y = [2.0, 2.1, 1.9, 5.0, 5.2, 10.1, 9.9, 17.1]

    table = make_table(x, y)

    result = clean_solve_quadratic_mls_table(table)

    assert result is not None

    actual_a, actual_b, actual_c = result
    expected_a, expected_b, expected_c = expected_quadratic_mls(x, y)

    assert actual_a == pytest.approx(expected_a, rel=1e-10, abs=1e-10)
    assert actual_b == pytest.approx(expected_b, rel=1e-10, abs=1e-10)
    assert actual_c == pytest.approx(expected_c, rel=1e-10, abs=1e-10)


def test_solve_quadratic_mls_table_noisy_parabola() -> None:
    x = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    y = [29.2, 17.7, 9.1, 3.4, 1.2, 2.1, 6.3, 13.6, 24.2]

    table = make_table(x, y)

    result = clean_solve_quadratic_mls_table(table)

    assert result is not None

    actual_a, actual_b, actual_c = result
    expected_a, expected_b, expected_c = expected_quadratic_mls(x, y)

    assert actual_a == pytest.approx(expected_a, rel=1e-10, abs=1e-10)
    assert actual_b == pytest.approx(expected_b, rel=1e-10, abs=1e-10)
    assert actual_c == pytest.approx(expected_c, rel=1e-10, abs=1e-10)


def test_solve_quadratic_mls_table_symmetric_parabola() -> None:
    x = [-3, -2, -1, 0, 1, 2, 3]
    y = [9, 4, 1, 0, 1, 4, 9]

    table = make_table(x, y)

    result = clean_solve_quadratic_mls_table(table)

    assert result is not None

    a, b, c = result

    assert a == pytest.approx(1.0)
    assert b == pytest.approx(0.0, abs=1e-12)
    assert c == pytest.approx(0.0, abs=1e-12)
