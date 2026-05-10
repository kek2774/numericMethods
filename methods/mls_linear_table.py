import pandas as pd
import numpy as np

from .methods_utils import norm


def compute_a_coef_linear(x: np.ndarray, y: np.ndarray) -> float:
    n: int = len(x)
    sum_x: float = sum(x)
    sum_y: float = sum(y)
    numerator: float = n * sum([x[i] * y[i] for i in range(n)]) - sum_x * sum_y
    denumerator: float = n * sum(x**2) - sum_x**2
    return numerator / denumerator


def compute_b_coef_linear(a: float, x: np.ndarray, y: np.ndarray) -> float:
    n: int = len(x)
    sum_y: float = sum(y)
    sum_x: float = sum(x)
    numerator: float = sum_y - a * sum_x
    denumerator: int = n
    return numerator / denumerator


def clean_solve_linear_mls_table(table: pd.DataFrame) -> tuple[float, float] | None:
    # Берем строку, преобразовываем ее в список, берем все элементы списка после первого (первый - название строки)
    try:
        x: np.ndarray = np.array(table.iloc[0].tolist()[1:], dtype=float)
        y: np.ndarray = np.array(table.iloc[1].tolist()[1:], dtype=float)
    except Exception:
        return None

    if len(x) == 0 or len(y) == 0:
        return None

    if len(x) != len(y):
        return None

    # Вычисляем оптимальные коэффициенты аппроксимации
    a: float = compute_a_coef_linear(x, y)
    b: float = compute_b_coef_linear(a, x, y)
    return a, b


def pretty_solve_linear_mls_table(table: pd.DataFrame) -> dict:
    # Берем строку, преобразовываем ее в список, берем все элементы списка после первого (первый - название строки)
    try:
        x: np.ndarray = np.array(table.iloc[0].tolist()[1:], dtype=float)
        y: np.ndarray = np.array(table.iloc[1].tolist()[1:], dtype=float)
    except Exception:
        return {
            "status": "error",
            "message": "Не удалось прочитать таблицу значений x и y",
        }

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
    a: float = compute_a_coef_linear(x, y)
    b: float = compute_b_coef_linear(a, x, y)

    y_pred: np.ndarray = np.array([a * x[i] + b for i in range(len(x))], dtype=float)
    residuals_vec: np.ndarray = abs(y_pred - y)

    return {
        "status": "ok",
        "a": a,
        "b": b,
        "y_pred": y_pred.copy(),
        "residuals": residuals_vec.copy(),
        "residuals_norm_inf": norm.vec_norm_inf(residuals_vec.copy()),
    }
