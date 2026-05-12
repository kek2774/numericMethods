import pandas as pd
import numpy as np


def clean_solve_diff_table(table: pd.DataFrame, x_dots: np.ndarray) -> float | None:

    # Берем строку, преобразовываем ее в список, берем все элементы списка после первого (первый - название строки)
    x: np.ndarray = np.array(table.iloc[0].tolist()[1:], dtype=float)
    y: np.ndarray = np.array(table.iloc[1].tolist()[1:], dtype=float)

    # Проверка на равномерную сетку
    if not all([x[i + 1] - x[i] == x[i + 2] - x[i + 1] for i in range(len(x) - 2)]):
        return None

    return None
