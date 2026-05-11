import numpy as np
import pandas as pd
import pytest

from methods.mls_linear_table import clean_solve_linear_mls_table


def make_table(x, y) -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["x", *x],
            ["y", *y],
        ]
    )


def expected_linear_mls(x, y) -> tuple[float, float]:
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)

    matrix = np.column_stack([x_arr, np.ones(len(x_arr))])
    a, b = np.linalg.lstsq(matrix, y_arr, rcond=None)[0]

    return float(a), float(b)


@pytest.mark.parametrize(
    "x, y",
    [
        (
            [-3, -2, -1, 0, 1, 2, 3],
            [-7, -5, -3, -1, 1, 3, 5],
        ),
        (
            [-10, -5, 0, 5, 10],
            [4, 4, 4, 4, 4],
        ),
        (
            [0.1, 0.2, 0.4, 0.8, 1.6],
            [1.12, 1.27, 1.55, 2.06, 2.91],
        ),
        (
            [5, -2, 10, 0, 3, -7],
            [15.2, -3.1, 28.4, 1.2, 9.3, -16.8],
        ),
        (
            [1, 1, 2, 2, 3, 3, 4, 4],
            [2.1, 1.9, 4.2, 3.8, 6.1, 5.9, 8.2, 7.8],
        ),
        (
            [-1000, -500, 0, 500, 1000],
            [-3001, -1501, -1, 1499, 2999],
        ),
        (
            [-2.5, -1.25, 0.0, 1.25, 2.5],
            [7.5, 2.1875, 1.0, 3.9375, 11.0],
        ),
        (
            [10, 20, 30, 40, 50, 60],
            [100.5, 98.1, 95.7, 94.2, 91.8, 89.5],
        ),
    ],
)
def test_solve_linear_mls_table_matches_lstsq_reference(
    x: list[float], y: list[float]
) -> None:
    table = make_table(x, y)

    result = clean_solve_linear_mls_table(table)

    assert result is not None

    actual_a, actual_b = result
    expected_a, expected_b = expected_linear_mls(x, y)

    assert actual_a == pytest.approx(expected_a, rel=1e-10, abs=1e-10)
    assert actual_b == pytest.approx(expected_b, rel=1e-10, abs=1e-10)


def test_solve_linear_mls_table_exact_line() -> None:
    x = [-4, -2, 0, 2, 4, 6]
    y = [-13, -7, -1, 5, 11, 17]

    table = make_table(x, y)

    result = clean_solve_linear_mls_table(table)

    assert result is not None

    a, b = result

    assert a == pytest.approx(3.0)
    assert b == pytest.approx(-1.0)


def test_solve_linear_mls_table_horizontal_line() -> None:
    x = [-100, -50, 0, 50, 100]
    y = [12.75, 12.75, 12.75, 12.75, 12.75]

    table = make_table(x, y)

    result = clean_solve_linear_mls_table(table)

    assert result is not None

    a, b = result

    assert a == pytest.approx(0.0, abs=1e-12)
    assert b == pytest.approx(12.75)


def test_solve_linear_mls_table_unsorted_x_values() -> None:
    x = [8, -3, 5, 0, 12, -7, 2]
    y = [19.3, -8.2, 12.1, 1.4, 28.7, -17.9, 5.8]

    table = make_table(x, y)

    result = clean_solve_linear_mls_table(table)

    assert result is not None

    actual_a, actual_b = result
    expected_a, expected_b = expected_linear_mls(x, y)

    assert actual_a == pytest.approx(expected_a, rel=1e-10, abs=1e-10)
    assert actual_b == pytest.approx(expected_b, rel=1e-10, abs=1e-10)


def test_solve_linear_mls_table_repeated_x_values_valid_case() -> None:
    x = [1, 1, 1, 2, 2, 3, 3, 4]
    y = [2.0, 2.1, 1.9, 4.0, 4.2, 6.1, 5.9, 8.1]

    table = make_table(x, y)

    result = clean_solve_linear_mls_table(table)

    assert result is not None

    actual_a, actual_b = result
    expected_a, expected_b = expected_linear_mls(x, y)

    assert actual_a == pytest.approx(expected_a, rel=1e-10, abs=1e-10)
    assert actual_b == pytest.approx(expected_b, rel=1e-10, abs=1e-10)


def test_solve_linear_mls_table_quadratic_data_linear_approximation() -> None:
    x = [-3, -2, -1, 0, 1, 2, 3]
    y = [9, 4, 1, 0, 1, 4, 9]

    table = make_table(x, y)

    result = clean_solve_linear_mls_table(table)

    assert result is not None

    a, b = result

    assert a == pytest.approx(0.0, abs=1e-12)
    assert b == pytest.approx(4.0)


def test_solve_linear_mls_table_non_symmetric_quadratic_data_linear_approximation() -> (
    None
):
    x = [0, 1, 2, 3, 4]
    y = [1, 2, 5, 10, 17]

    table = make_table(x, y)

    result = clean_solve_linear_mls_table(table)

    assert result is not None

    actual_a, actual_b = result
    expected_a, expected_b = expected_linear_mls(x, y)

    assert actual_a == pytest.approx(expected_a, rel=1e-10, abs=1e-10)
    assert actual_b == pytest.approx(expected_b, rel=1e-10, abs=1e-10)
