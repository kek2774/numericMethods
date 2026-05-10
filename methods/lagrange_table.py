import pandas as pd
import numpy as np


def small_l(x_arr: np.ndarray, x_fixed: float, j: int) -> float:
    prod = 1
    for i in range(len(x_arr)):
        if i == j:
            continue
        prod *= (x_fixed - x_arr[i]) / (x_arr[j] - x_arr[i])

    return prod


def clean_solve_lagrange_table(table: pd.DataFrame, x_star: float) -> float | None:

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


def pretty_solve_lagrange_table(table: pd.DataFrame, x_star: float) -> dict:

    try:
        x: np.ndarray = np.array(table.iloc[0].tolist()[1:], dtype=float)
        y: np.ndarray = np.array(table.iloc[1].tolist()[1:], dtype=float)
    except Exception:
        return {
            "status": "error",
            "message": "Не удалось прочитать таблицу значений x и y.",
        }

    if len(x) == 0 or len(y) == 0:
        return {
            "status": "error",
            "message": "Таблица значений пуста.",
        }

    if len(x) != len(y):
        return {
            "status": "error",
            "message": "Количество значений x и y не совпадает.",
        }

    if len(np.unique(x)) != len(x):
        return {
            "status": "error",
            "message": "Значения x должны быть различными.",
        }

    if not isinstance(x_star, (float, int)):
        return {
            "status": "error",
            "message": "x* должен быть числом.",
        }

    rows: list = []
    factor_rows: list = []

    L_n: float = 0.0

    for j in range(len(x)):
        l_j = 1.0

        for i in range(len(x)):
            if i == j:
                continue

            numerator: float = x_star - x[i]
            denominator: float = x[j] - x[i]
            factor: float = numerator / denominator

            l_j *= factor

            factor_rows.append(
                {
                    "j": j + 1,
                    "i": i + 1,
                    "x*": x_star,
                    "x_i": x[i],
                    "x_j": x[j],
                    "x* - x_i": numerator,
                    "x_j - x_i": denominator,
                    "factor": factor,
                }
            )

        prod: float = y[j] * l_j
        L_n += prod

        rows.append(
            {
                "j": j + 1,
                "x_j": x[j],
                "y_j": y[j],
                "l_j(x*)": l_j,
                "y_j * l_j(x*)": prod,
            }
        )

    terms_df = pd.DataFrame(rows)
    factors_df = pd.DataFrame(factor_rows)

    return {
        "status": "ok",
        "x": x.copy(),
        "y": y.copy(),
        "x_star": float(x_star),
        "result": float(L_n),
        "terms_table": terms_df,
        "factors_table": factors_df,
    }
