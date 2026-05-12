import pandas as pd
import numpy as np


def clean_solve_first_diff_table(table: pd.DataFrame, x_dots: np.ndarray) -> np.ndarray:

    _ = x_dots

    # Берем строку, преобразовываем ее в список, берем все элементы списка после первого (первый - название строки)
    x: np.ndarray = np.array(table.iloc[0].tolist()[1:], dtype=float)
    y: np.ndarray = np.array(table.iloc[1].tolist()[1:], dtype=float)

    uniform_grid_flag: bool = True

    # Проверка на равномерную сетку
    if not all([x[i + 1] - x[i] == x[i + 2] - x[i + 1] for i in range(len(x) - 2)]):
        uniform_grid_flag = False

    n: int = len(x)
    nodes_count: int = n - 1
    y_1_diffs: np.ndarray = np.zeros(n, dtype=float)

    if uniform_grid_flag:
        for i in range(1, nodes_count):
            h = x[i + 1] - x[i]
            delta = y[i + 1] - y[i - 1]
            y_1_diffs[i] = delta / (2 * h)

    else:
        for i in range(nodes_count):
            h = x[i + 1] - x[i]
            delta = y[i + 1] - y[i]
            y_1_diffs[i] = delta / h

    return y_1_diffs
