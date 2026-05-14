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


def pretty_solve_first_diff_table(table: pd.DataFrame, x_dots: np.ndarray) -> dict:

    _ = x_dots

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
            "message": "Таблица значений пустая",
        }

    if len(x) != len(y):
        return {
            "status": "error",
            "message": "Количество значений x и y не совпадает",
        }

    if len(x) < 2:
        return {
            "status": "error",
            "message": "Для первой производной нужно минимум 2 точки",
        }

    if len(np.unique(x)) != len(x):
        return {
            "status": "error",
            "message": "Значения x не должны повторяться",
            "x": x.copy(),
            "y": y.copy(),
        }

    uniform_grid_flag: bool = True

    if len(x) > 2:
        uniform_grid_flag = all(
            [
                np.isclose(x[i + 1] - x[i], x[i + 2] - x[i + 1])
                for i in range(len(x) - 2)
            ]
        )

    n: int = len(x)
    nodes_count: int = n - 1
    y_1_diffs: np.ndarray = np.zeros(n, dtype=float)
    rows: list[dict] = []

    if uniform_grid_flag:
        for i in range(1, nodes_count):
            h: float = x[i + 1] - x[i]
            delta: float = y[i + 1] - y[i - 1]
            y_1_diffs[i] = delta / (2 * h)

            rows.append(
                {
                    "i": i,
                    "x_i": x[i],
                    "formula": "(y_i+1 - y_i-1) / (2h)",
                    "h": h,
                    "delta": delta,
                    "y'_i": y_1_diffs[i],
                }
            )

    else:
        for i in range(nodes_count):
            h: float = x[i + 1] - x[i]
            delta: float = y[i + 1] - y[i]
            y_1_diffs[i] = delta / h

            rows.append(
                {
                    "i": i,
                    "x_i": x[i],
                    "formula": "(y_i+1 - y_i) / h_i",
                    "h_i": h,
                    "delta": delta,
                    "y'_i": y_1_diffs[i],
                }
            )

    return {
        "status": "ok",
        "x": x.copy(),
        "y": y.copy(),
        "x_dots": np.array(x_dots, dtype=float).copy(),
        "result": y_1_diffs.copy(),
        "uniform_grid": uniform_grid_flag,
        "diff_table": pd.DataFrame(rows),
    }
