import numpy as np
import pytest

from methods.mpi_matrix import clean_solve_mpi_matrix


def assert_solver_result(result, A, b, expected_x=None, tol=1e-6):
    assert result is not None, "Функция вернула None, хотя ожидалось решение"
    assert isinstance(
        result, tuple
    ), "Функция должна вернуть tuple: (solution, residual)"
    assert (
        len(result) == 2
    ), "Функция должна вернуть ровно два значения: (solution, residual)"

    x, residual = result

    assert isinstance(x, np.ndarray), "Первый элемент должен быть np.ndarray с решением"
    assert isinstance(
        residual, np.ndarray
    ), "Второй элемент должен быть np.ndarray с невязкой"

    assert x.shape == b.shape, "Размер решения должен совпадать с размером b"
    assert residual.shape == b.shape, "Размер невязки должен совпадать с размером b"

    if expected_x is not None:
        assert np.allclose(x, expected_x, atol=tol), (
            f"\nОжидалось решение: {expected_x}" f"\nПолучено решение: {x}"
        )

    true_residual = b - A @ x

    assert np.allclose(true_residual, 0, atol=tol), (
        f"\nРешение не удовлетворяет системе Ax = b" f"\nb - A @ x = {true_residual}"
    )

    assert np.allclose(residual, 0, atol=tol), (
        f"\nВозвращённая невязка должна быть близка к нулю" f"\nПолучено: {residual}"
    )


def test_identity_matrix():
    A = np.eye(4)
    b = np.array([1.0, -2.0, 3.5, 10.0])

    result = clean_solve_mpi_matrix(1e-10, A, b)

    expected_x = b
    assert_solver_result(result, A, b, expected_x)


def test_diagonal_matrix():
    A = np.array(
        [
            [2.0, 0.0, 0.0],
            [0.0, -4.0, 0.0],
            [0.0, 0.0, 5.0],
        ]
    )
    b = np.array([8.0, 12.0, 20.0])

    result = clean_solve_mpi_matrix(1e-10, A, b)

    expected_x = np.array([4.0, -3.0, 4.0])
    assert_solver_result(result, A, b, expected_x)


def test_simple_diagonally_dominant_2x2_system():
    A = np.array(
        [
            [4.0, 1.0],
            [2.0, 5.0],
        ]
    )
    b = np.array([9.0, 12.0])

    result = clean_solve_mpi_matrix(1e-10, A, b)

    expected_x = np.linalg.solve(A, b)
    assert_solver_result(result, A, b, expected_x)


def test_simple_diagonally_dominant_3x3_system():
    A = np.array(
        [
            [10.0, 1.0, 1.0],
            [2.0, 10.0, 1.0],
            [2.0, 2.0, 10.0],
        ]
    )
    b = np.array([12.0, 13.0, 14.0])

    result = clean_solve_mpi_matrix(1e-10, A, b)

    expected_x = np.linalg.solve(A, b)
    assert_solver_result(result, A, b, expected_x)


def test_requires_row_swap_should_return_none_or_solve():
    """
    Для обычного МПИ без перестановок строк такой случай часто считается неподходящим,
    потому что на диагонали стоит ноль.
    Если твоя функция умеет переставлять строки, она может вернуть решение.
    """
    A = np.array(
        [
            [0.0, 2.0],
            [1.0, 3.0],
        ]
    )
    b = np.array([4.0, 7.0])

    result = clean_solve_mpi_matrix(1e-10, A, b)

    if result is None:
        assert result is None
    else:
        expected_x = np.linalg.solve(A, b)
        assert_solver_result(result, A, b, expected_x)


@pytest.mark.parametrize(
    "A,b",
    [
        (
            np.array(
                [
                    [5.0, 1.0],
                    [1.0, 4.0],
                ]
            ),
            np.array([6.0, 5.0]),
        ),
        (
            np.array(
                [
                    [8.0, -1.0, 1.0],
                    [2.0, 9.0, -1.0],
                    [1.0, -1.0, 7.0],
                ]
            ),
            np.array([8.0, 9.0, 7.0]),
        ),
        (
            np.array(
                [
                    [12.0, 2.0, -1.0],
                    [1.0, 10.0, 2.0],
                    [-1.0, 2.0, 9.0],
                ]
            ),
            np.array([15.0, 13.0, 10.0]),
        ),
    ],
)
def test_parametrized_diagonally_dominant_systems(A, b):
    result = clean_solve_mpi_matrix(1e-10, A, b)

    expected_x = np.linalg.solve(A, b)
    assert_solver_result(result, A, b, expected_x)


@pytest.mark.parametrize("n", [2, 3, 5, 10])
def test_random_diagonally_dominant_systems(n):
    rng = np.random.default_rng(42)

    A = rng.uniform(-2.0, 2.0, size=(n, n))

    for i in range(n):
        row_sum = np.sum(np.abs(A[i])) - abs(A[i, i])
        A[i, i] = row_sum + rng.uniform(2.0, 5.0)

    expected_x = rng.normal(size=n)
    b = A @ expected_x

    result = clean_solve_mpi_matrix(1e-10, A, b, maxiter=10000)

    assert_solver_result(result, A, b, expected_x, tol=1e-6)


def test_input_arrays_are_not_modified():
    A = np.array(
        [
            [4.0, 1.0],
            [1.0, 3.0],
        ]
    )
    b = np.array([5.0, 6.0])

    A_original = A.copy()
    b_original = b.copy()

    clean_solve_mpi_matrix(1e-10, A, b)

    assert np.array_equal(A, A_original), "Функция изменила A"
    assert np.array_equal(b, b_original), "Функция изменила b"


def test_singular_matrix_returns_none():
    A = np.array(
        [
            [1.0, 2.0],
            [2.0, 4.0],
        ]
    )
    b = np.array([3.0, 6.0])

    result = clean_solve_mpi_matrix(1e-10, A, b)

    assert result is None


def test_zero_diagonal_returns_none():
    A = np.array(
        [
            [0.0, 1.0],
            [1.0, 3.0],
        ]
    )
    b = np.array([1.0, 4.0])

    result = clean_solve_mpi_matrix(1e-10, A, b)

    assert result is None
