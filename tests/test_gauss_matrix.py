import numpy as np
import pytest

from methods.gauss_matrix import clean_solve_gauss_matrix


def assert_solver_result(result, A, b, expected_x=None, tol=1e-8):
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

    true_residual = A @ x - b

    assert np.allclose(true_residual, 0, atol=tol), (
        f"\nРешение не удовлетворяет системе Ax = b" f"\nA @ x - b = {true_residual}"
    )

    assert np.allclose(residual, 0, atol=tol), (
        f"\nВозвращённая невязка должна быть близка к нулю" f"\nПолучено: {residual}"
    )


def test_simple_2x2_system():
    A = np.array(
        [
            [2.0, 1.0],
            [1.0, -1.0],
        ]
    )
    b = np.array([5.0, 1.0])

    result = clean_solve_gauss_matrix(A, b)

    expected_x = np.array([2.0, 1.0])
    assert_solver_result(result, A, b, expected_x)


def test_simple_3x3_system():
    A = np.array(
        [
            [2.0, 1.0, -1.0],
            [-3.0, -1.0, 2.0],
            [-2.0, 1.0, 2.0],
        ]
    )
    b = np.array([8.0, -11.0, -3.0])

    result = clean_solve_gauss_matrix(A, b)

    expected_x = np.array([2.0, 3.0, -1.0])
    assert_solver_result(result, A, b, expected_x)


def test_identity_matrix():
    A = np.eye(4)
    b = np.array([1.0, -2.0, 3.5, 10.0])

    result = clean_solve_gauss_matrix(A, b)

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

    result = clean_solve_gauss_matrix(A, b)

    expected_x = np.array([4.0, -3.0, 4.0])
    assert_solver_result(result, A, b, expected_x)


def test_requires_row_swap():
    """
    Проверяет случай, где первый ведущий элемент равен нулю.
    Нормальный метод Гаусса должен переставить строки.
    """
    A = np.array(
        [
            [0.0, 2.0],
            [1.0, 3.0],
        ]
    )
    b = np.array([4.0, 7.0])

    result = clean_solve_gauss_matrix(A, b)

    expected_x = np.array([1.0, 2.0])
    assert_solver_result(result, A, b, expected_x)


def test_upper_triangular_matrix():
    A = np.array(
        [
            [2.0, -1.0, 1.0],
            [0.0, 3.0, -2.0],
            [0.0, 0.0, 4.0],
        ]
    )
    b = np.array([1.0, 7.0, 8.0])

    result = clean_solve_gauss_matrix(A, b)

    expected_x = np.linalg.solve(A, b)
    assert_solver_result(result, A, b, expected_x)


def test_compare_with_numpy_solve():
    A = np.array(
        [
            [10.0, -2.0, 3.0],
            [1.0, 5.0, -1.0],
            [2.0, 3.0, 8.0],
        ]
    )
    b = np.array([7.0, -8.0, 6.0])

    result = clean_solve_gauss_matrix(A, b)

    expected_x = np.linalg.solve(A, b)
    assert_solver_result(result, A, b, expected_x)


@pytest.mark.parametrize(
    "A,b",
    [
        (
            np.array(
                [
                    [4.0, 2.0],
                    [1.0, 3.0],
                ]
            ),
            np.array([10.0, 8.0]),
        ),
        (
            np.array(
                [
                    [1.0, 1.0, 1.0],
                    [2.0, -1.0, 1.0],
                    [1.0, 2.0, -1.0],
                ]
            ),
            np.array([6.0, 3.0, 2.0]),
        ),
        (
            np.array(
                [
                    [3.0, -1.0, 2.0],
                    [1.0, 4.0, -2.0],
                    [2.0, 1.0, 5.0],
                ]
            ),
            np.array([10.0, -1.0, 12.0]),
        ),
    ],
)
def test_parametrized_systems(A, b):
    result = clean_solve_gauss_matrix(A, b)

    expected_x = np.linalg.solve(A, b)
    assert_solver_result(result, A, b, expected_x)


def test_random_well_conditioned_systems():
    rng = np.random.default_rng(42)

    for n in range(2, 8):
        A = rng.normal(size=(n, n))

        # Делаем матрицу лучше обусловленной,
        # чтобы тест не падал из-за численной неустойчивости.
        A += np.eye(n) * n

        expected_x = rng.normal(size=n)
        b = A @ expected_x

        result = clean_solve_gauss_matrix(A, b)

        assert_solver_result(result, A, b, expected_x, tol=1e-7)


def test_input_arrays_are_not_modified():
    """
    Функция не должна портить исходные A и b.
    Если твоя реализация специально меняет массивы inplace,
    этот тест можно удалить.
    """
    A = np.array(
        [
            [3.0, 2.0],
            [1.0, 2.0],
        ]
    )
    b = np.array([5.0, 5.0])

    A_original = A.copy()
    b_original = b.copy()

    clean_solve_gauss_matrix(A, b)

    assert np.array_equal(A, A_original), "Функция изменила A"
    assert np.array_equal(b, b_original), "Функция изменила b"


def test_singular_matrix_with_infinite_solutions_returns_none():
    """
    Вторая строка кратна первой.
    Система имеет бесконечно много решений.
    Единственного решения нет.
    """
    A = np.array(
        [
            [1.0, 2.0],
            [2.0, 4.0],
        ]
    )
    b = np.array([3.0, 6.0])

    result = clean_solve_gauss_matrix(A, b)

    assert result is None


def test_inconsistent_system_returns_none():
    """
    Система несовместна:
    x + 2y = 3
    2x + 4y = 7
    """
    A = np.array(
        [
            [1.0, 2.0],
            [2.0, 4.0],
        ]
    )
    b = np.array([3.0, 7.0])

    result = clean_solve_gauss_matrix(A, b)

    assert result is None


def test_zero_matrix_returns_none():
    A = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    b = np.array([0.0, 0.0])

    result = clean_solve_gauss_matrix(A, b)

    assert result is None
