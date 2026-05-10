import numpy as np
import pandas as pd
from .methods_utils import matrix_diag


def P_n(x_fixed: float, a: np.ndarray, x_arr: np.ndarray, n: int) -> float:
    summa: float = 0
    for i in range(n):
        prod = 1
        for j in range(i):
            prod *= x_fixed - x_arr[j]
        summa += prod * a[i]

    return summa


def get_divided_difference_matrix(x_arr: np.ndarray, y_arr: np.ndarray) -> np.ndarray:
    n: int = len(x_arr)
    div_dif_matrix: np.ndarray = np.full((n, n + 1), None, dtype=float)
    div_dif_matrix = div_dif_matrix.T
    div_dif_matrix[0] = x_arr.copy()
    div_dif_matrix[1] = y_arr.copy()

    for i in range(2, n + 1):
        order: int = i - 1
        for j in range(order, n):
            numerator: float = div_dif_matrix[i - 1, j] - div_dif_matrix[i - 1, j - 1]
            denumerator: float = div_dif_matrix[0, j] - div_dif_matrix[0, j - order]
            div_dif_matrix[i, j] = numerator / denumerator

    return div_dif_matrix


def clean_solve_newton_table(table: pd.DataFrame, x_star) -> float | None:

    # Берем строку, преобразовываем ее в список, берем все элементы списка после первого (первый - название строки)
    x: np.ndarray = np.array(table.iloc[0].tolist()[1:], dtype=float)
    y: np.ndarray = np.array(table.iloc[1].tolist()[1:], dtype=float)

    if len(np.unique(x)) != len(x):
        return None

    if len(x) != len(y):
        return None

    if len(x) == 0 or len(y) == 0:
        return None

    if not isinstance(x_star, (float, int)):
        return None

    div_diffs: np.ndarray = get_divided_difference_matrix(x, y)
    a: np.ndarray = matrix_diag.get_diag_as_vector(div_diffs[1:, :])

    return P_n(x_star, a, x, a.size)


def pretty_solve_newton_table(table: pd.DataFrame, x_star) -> dict:

    # Берем строку, преобразовываем ее в список, берем все элементы списка после первого (первый - название строки)
    try:
        x: np.ndarray = np.array(table.iloc[0].tolist()[1:], dtype=float)
        y: np.ndarray = np.array(table.iloc[1].tolist()[1:], dtype=float)
        x_star = float(x_star)
    except Exception:
        return {
            "status": "error",
            "message": "Не удалось прочитать таблицу значений x и y или x_star",
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

    if len(np.unique(x)) != len(x):
        return {
            "status": "error",
            "message": "Значения x должны быть различны",
        }

    div_diffs: np.ndarray = get_divided_difference_matrix(x, y)
    a: np.ndarray = matrix_diag.get_diag_as_vector(div_diffs[1:, :])

    result: float = P_n(x_star, a, x, a.size)

    # Таблица коэффициентов a_i
    coef_rows = []
    for i in range(len(a)):
        coef_rows.append(
            {
                "i": i,
                "a_i": a[i],
            }
        )

    coefficients_table = pd.DataFrame(coef_rows)

    # Таблица слагаемых полинома Ньютона
    term_rows = []
    cumulative_sum = 0.0

    for i in range(len(a)):
        prod = 1.0

        for j in range(i):
            prod *= x_star - x[j]

        term = prod * a[i]
        cumulative_sum += term

        term_rows.append(
            {
                "i": i,
                "a_i": a[i],
                "product": prod,
                "term": term,
                "partial_sum": cumulative_sum,
            }
        )

    terms_table = pd.DataFrame(term_rows)

    return {
        "status": "ok",
        "x": x.copy(),
        "y": y.copy(),
        "x_star": x_star,
        "result": float(result),
        "divided_differences_matrix": div_diffs.copy(),
        "coefficients": a.copy(),
        "coefficients_table": coefficients_table,
        "terms_table": terms_table,
    }
