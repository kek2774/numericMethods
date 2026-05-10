import pandas as pd
import numpy as np


def small_l(x_arr: np.ndarray, x_fixed: float, j: int) -> float:
    prod = 1
    for i in range(len(x_arr)):
        if i == j:
            continue
        prod *= (x_fixed - x_arr[i]) / (x_arr[j] - x_arr[i])

    return prod


def solve_lagrange_table(table: pd.DataFrame, x_star: float) -> float | None:

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

    L_n: float = sum([small_l(x, x_star, j) * y[j] for j in range(len(x))])

    return L_n
