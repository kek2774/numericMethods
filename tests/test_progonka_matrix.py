import numpy as np
import pytest

from methods.progonka_matrix import clean_solve_progonka_matrix


def assert_solver_result(result, A, b, expected_x=None, tol=1e-8):
    assert result is not None, "Функция вернула None, хотя ожидалось решение"
    assert isinstance(
        result, tuple
    ), "Функция должна вернуть tuple: (solution, residual)"
    assert (
        len(result) == 2
    ), "Функция должна вернуть ровно два значения: (solution, residual)"

    x, residual = result

    assert isinstance(
        x, np.ndarray
    ), "Первый элемент результата должен быть np.ndarray с решением"
    assert isinstance(
        residual, np.ndarray
    ), "Второй элемент результата должен быть np.ndarray с невязкой"

    assert x.shape == b.shape, "Размер решения должен совпадать с размером b"
    assert residual.shape == b.shape, "Размер невязки должен совпадать с размером b"

    if expected_x is not None:
        assert np.allclose(x, expected_x, atol=tol), (
            f"\nОжидалось решение: {expected_x}" f"\nПолучено решение: {x}"
        )

    calculated_residual = A @ x - b

    assert np.allclose(calculated_residual, 0, atol=tol), (
        f"\nРешение не удовлетворяет системе Ax = b"
        f"\nA @ x - b = {calculated_residual}"
    )

    assert np.allclose(residual, 0, atol=tol), (
        f"\nНевязка должна быть близка к нулю" f"\nПолучено: {residual}"
    )


def test_simple_3x3_tridiagonal_system():
    A = np.array(
        [
            [2.0, -1.0, 0.0],
            [-1.0, 2.0, -1.0],
            [0.0, -1.0, 2.0],
        ]
    )
    b = np.array([1.0, 0.0, 1.0])

    result = clean_solve_progonka_matrix(A, b)

    expected_x = np.array([1.0, 1.0, 1.0])
    assert_solver_result(result, A, b, expected_x)


def test_4x4_tridiagonal_system():
    A = np.array(
        [
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 4.0, 1.0, 0.0],
            [0.0, 1.0, 4.0, 1.0],
            [0.0, 0.0, 1.0, 4.0],
        ]
    )

    expected_x = np.array([1.0, 2.0, 3.0, 4.0])
    b = A @ expected_x

    result = clean_solve_progonka_matrix(A, b)

    assert_solver_result(result, A, b, expected_x)


def test_compare_with_numpy_solve():
    A = np.array(
        [
            [10.0, -2.0, 0.0, 0.0],
            [3.0, 10.0, -4.0, 0.0],
            [0.0, 1.0, 8.0, -2.0],
            [0.0, 0.0, -1.0, 5.0],
        ]
    )
    b = np.array([6.0, 25.0, -11.0, -11.0])

    result = clean_solve_progonka_matrix(A, b)

    expected_x = np.linalg.solve(A, b)
    assert_solver_result(result, A, b, expected_x)


def test_identity_matrix():
    A = np.eye(5)
    b = np.array([1.0, -2.0, 3.0, 4.5, 10.0])

    result = clean_solve_progonka_matrix(A, b)

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

    result = clean_solve_progonka_matrix(A, b)

    expected_x = np.array([4.0, -3.0, 4.0])
    assert_solver_result(result, A, b, expected_x)


@pytest.mark.parametrize("n", [2, 3, 5, 10])
def test_random_diagonally_dominant_tridiagonal_systems(n):
    rng = np.random.default_rng(42)

    lower = rng.uniform(-2.0, 2.0, size=n - 1)
    upper = rng.uniform(-2.0, 2.0, size=n - 1)

    main = np.empty(n)

    for i in range(n):
        left = abs(lower[i - 1]) if i > 0 else 0.0
        right = abs(upper[i]) if i < n - 1 else 0.0
        main[i] = left + right + rng.uniform(1.0, 5.0)

    A = np.zeros((n, n))

    for i in range(n):
        A[i, i] = main[i]

    for i in range(n - 1):
        A[i + 1, i] = lower[i]
        A[i, i + 1] = upper[i]

    expected_x = rng.normal(size=n)
    b = A @ expected_x

    result = clean_solve_progonka_matrix(A, b)

    assert_solver_result(result, A, b, expected_x, tol=1e-7)


def test_input_arrays_are_not_modified():
    A = np.array(
        [
            [4.0, 1.0, 0.0],
            [1.0, 4.0, 1.0],
            [0.0, 1.0, 4.0],
        ]
    )
    b = np.array([6.0, 12.0, 14.0])

    A_original = A.copy()
    b_original = b.copy()

    clean_solve_progonka_matrix(A, b)

    assert np.array_equal(A, A_original), "Функция изменила A"
    assert np.array_equal(b, b_original), "Функция изменила b"


def test_singular_matrix_returns_none():
    A = np.array(
        [
            [0.0, 1.0],
            [1.0, 2.0],
        ]
    )
    b = np.array([1.0, 2.0])

    result = clean_solve_progonka_matrix(A, b)

    assert result is None
