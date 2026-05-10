import pandas as pd
import numpy as np

from .methods_utils import norm


def compute_denumerator_quadratic(x: np.ndarray) -> float:
    n: int = len(x)
    x_sums: list[float] = [float(sum(x**i)) for i in range(1, 5)]

    denumerator_part1: float = x_sums[3] * (x_sums[1] * n - x_sums[0] ** 2)
    denumerator_part2: float = -x_sums[2] * (x_sums[2] * n - x_sums[0] * x_sums[1])
    denumerator_part3: float = x_sums[1] * (x_sums[2] * x_sums[0] - x_sums[1] ** 2)

    denumerator: float = denumerator_part1 + denumerator_part2 + denumerator_part3
    return denumerator


def compute_a_coef_quadratic(x: np.ndarray, y: np.ndarray) -> float:
    n: int = len(x)
    x_sums: list[float] = [float(sum(x**i)) for i in range(1, 5)]

    y_sum: float = float(sum(y))
    x_prod_y_sum: float = float(sum([x[i] * y[i] for i in range(n)]))
    x_squared_prod_y_sum: float = float(sum([x[i] ** 2 * y[i] for i in range(n)]))

    numerator_part1: float = x_squared_prod_y_sum * (x_sums[1] * n - x_sums[0] ** 2)
    numerator_part2: float = -x_sums[2] * (x_prod_y_sum * n - x_sums[0] * y_sum)
    numerator_part3: float = x_sums[1] * (x_prod_y_sum * x_sums[0] - x_sums[1] * y_sum)

    numerator: float = numerator_part1 + numerator_part2 + numerator_part3
    denumerator: float = compute_denumerator_quadratic(x)

    return numerator / denumerator


def compute_b_coef_quadratic(x: np.ndarray, y: np.ndarray) -> float:
    n: int = len(x)
    x_sums: list[float] = [float(sum(x**i)) for i in range(1, 5)]

    y_sum: float = float(sum(y))
    x_prod_y_sum: float = float(sum([x[i] * y[i] for i in range(n)]))
    x_squared_prod_y_sum: float = float(sum([x[i] ** 2 * y[i] for i in range(n)]))

    numerator_part1: float = x_sums[3] * (x_prod_y_sum * n - x_sums[0] * y_sum)
    numerator_part2: float = -x_squared_prod_y_sum * (
        x_sums[2] * n - x_sums[0] * x_sums[1]
    )
    numerator_part3: float = x_sums[1] * (x_sums[2] * y_sum - x_prod_y_sum * x_sums[1])

    numerator: float = numerator_part1 + numerator_part2 + numerator_part3
    denumerator: float = compute_denumerator_quadratic(x)

    return numerator / denumerator


def compute_c_coef_quadratic(x: np.ndarray, y: np.ndarray) -> float:
    n: int = len(x)
    x_sums: list[float] = [float(sum(x**i)) for i in range(1, 5)]

    y_sum: float = float(sum(y))
    x_prod_y_sum: float = float(sum([x[i] * y[i] for i in range(n)]))
    x_squared_prod_y_sum: float = float(sum([x[i] ** 2 * y[i] for i in range(n)]))

    numerator_part1: float = x_sums[3] * (x_sums[1] * y_sum - x_prod_y_sum * x_sums[0])
    numerator_part2: float = -x_sums[2] * (x_sums[2] * y_sum - x_prod_y_sum * x_sums[1])
    numerator_part3: float = x_squared_prod_y_sum * (
        x_sums[2] * x_sums[0] - x_sums[1] ** 2
    )

    numerator: float = numerator_part1 + numerator_part2 + numerator_part3
    denumerator: float = compute_denumerator_quadratic(x)

    return numerator / denumerator


def clean_solve_quadratic_mls_table(
    table: pd.DataFrame,
) -> tuple[float, float, float] | None:
    # Берем строку, преобразовываем ее в список, берем все элементы списка после первого (первый - название строки)
    x: np.ndarray = np.array(table.iloc[0].tolist()[1:], dtype=float)
    y: np.ndarray = np.array(table.iloc[1].tolist()[1:], dtype=float)

    if len(x) != len(y):
        return None

    if len(x) == 0 or len(y) == 0:
        return None

    # Вычисляем оптимальные коэффициенты аппроксимации
    a: float = compute_a_coef_quadratic(x, y)
    b: float = compute_b_coef_quadratic(x, y)
    c: float = compute_c_coef_quadratic(x, y)
    return a, b, c


def pretty_solve_quadratic_mls_table(
    table: pd.DataFrame,
) -> dict:
    # Берем строку, преобразовываем ее в список, берем все элементы списка после первого (первый - название строки)
    x: np.ndarray = np.array(table.iloc[0].tolist()[1:], dtype=float)
    y: np.ndarray = np.array(table.iloc[1].tolist()[1:], dtype=float)

    if len(x) == 0 or len(y) == 0:
        return {
            "status": "error",
            "message": "Таблица значений пуста",
        }

    if len(x) != len(y):
        return {
            "status": "error",
            "message": "Количество значений x и y не совпадает",
        }

    # Вычисляем оптимальные коэффициенты аппроксимации
    a: float = compute_a_coef_quadratic(x, y)
    b: float = compute_b_coef_quadratic(x, y)
    c: float = compute_c_coef_quadratic(x, y)

    y_pred: np.ndarray = np.array(
        [x[i] ** 2 * a + x[i] * b + c for i in range(len(x))], dtype=float
    )
    residuals_vec: np.ndarray = abs(y - y_pred)

    return {
        "status": "ok",
        "a": a,
        "b": b,
        "c": c,
        "y_pred": y_pred.copy(),
        "residuals": residuals_vec.copy(),
        "residuals_norm_inf": norm.vec_norm_inf(residuals_vec.copy()),
    }
