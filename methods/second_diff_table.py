import pandas as pd
import numpy as np


def clean_solve_second_diff_table(
    table: pd.DataFrame, x_dots: np.ndarray
) -> np.ndarray | None:
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
    y_2_diffs: np.ndarray = np.zeros(n - 2, dtype=float)

    if uniform_grid_flag:
        for i in range(1, nodes_count):
            h = x[i] - x[i - 1]
            delta = y[i + 1] - 2 * y[i] + y[i - 1]
            y_2_diffs[i - 1] = delta / (h**2)

    else:
        return None

    return y_2_diffs
