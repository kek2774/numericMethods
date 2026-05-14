import numpy as np
import pandas as pd
from typing import Any

from .progonka_matrix import clean_solve_progonka_matrix
from .methods_utils import matrix_diag


def get_h_and_deltas(x: np.ndarray, y: np.ndarray) -> tuple | None:
    result: tuple = tuple()
    n: int = len(x)
    nodes_count: int = n - 1

    h: np.ndarray = np.zeros(nodes_count, dtype=float)
    deltas: np.ndarray = np.zeros(nodes_count, dtype=float)
    for i in range(nodes_count):
        h[i] = x[i + 1] - x[i]

        if h[i] == 0:
            return None

        deltas[i] = (y[i + 1] - y[i]) / h[i]

    result = tuple([h, deltas])
    return result


def get_cubic_spline_coeffs(x: np.ndarray, y: np.ndarray) -> dict | None:
    coeffs: dict = {"a": [], "b": [], "c": [], "d": []}
    n: int = len(x)
    nodes_count = n - 1
    c: np.ndarray | None = get_c_coeffs(x, y)
    if c is None:
        return None

    a: np.ndarray = np.zeros(nodes_count, dtype=float)
    d: np.ndarray = np.zeros(nodes_count, dtype=float)
    b: np.ndarray = np.zeros(nodes_count, dtype=float)
    h_func_res: tuple | None = get_h_and_deltas(x, y)
    if h_func_res is None:
        return None

    h, deltas = h_func_res

    for i in range(nodes_count):
        a[i] = y[i]
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])
        b[i] = deltas[i] - h[i] / 3 * (2 * c[i] + c[i + 1])

    coeffs["a"] = a
    coeffs["b"] = b
    coeffs["c"] = c
    coeffs["d"] = d

    return coeffs


def get_c_coeffs(x: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    n: int = len(x)
    nodes_count: int = n - 1
    A_matrix: np.ndarray = np.zeros((nodes_count - 1, nodes_count - 1), dtype=float)
    b_vec: np.ndarray = np.zeros(nodes_count - 1, dtype=float)

    c: np.ndarray = np.zeros(n, dtype=float)

    h_func_res: tuple | None = get_h_and_deltas(x, y)
    if h_func_res is None:
        return None

    h, deltas = h_func_res

    for i in range(1, n - 1):
        row: int = i - 1
        a: float = h[i - 1]
        b: float = 2 * (h[i - 1] + h[i])
        c_tmp: float = h[i]
        A_matrix[row] = matrix_diag.build_diag_row(a, b, c_tmp, A_matrix.shape[0], row)
        b_vec[row] = 3 * (deltas[i] - deltas[i - 1])

    if nodes_count == 1:
        res = (np.array([], dtype=float), np.array([], dtype=float))
    else:
        res = clean_solve_progonka_matrix(A_matrix, b_vec)
    if res is None:
        return None

    if c[0] != c[-1]:
        return None

    c[1:nodes_count] = np.array(res[0], dtype=float)

    return c


def clean_solve_cubic_splines_table(table: pd.DataFrame) -> list | None:

    # Берем строку, преобразовываем ее в список, берем все элементы списка после первого (первый - название строки)
    x: np.ndarray = np.array(table.iloc[0].tolist()[1:], dtype=float)
    y: np.ndarray = np.array(table.iloc[1].tolist()[1:], dtype=float)

    if len(np.unique(x)) != len(x):
        return None

    if len(x) != len(y):
        return None

    if len(x) < 2 or len(y) < 2:
        return None

    if len(np.unique(x)) != len(x):
        return None

    cubic_coeffs: dict | None = get_cubic_spline_coeffs(x, y)
    if cubic_coeffs is None:
        return None

    result_list: list = list()
    result_list.append(cubic_coeffs["a"])
    result_list.append(cubic_coeffs["b"])
    result_list.append(cubic_coeffs["c"])
    result_list.append(cubic_coeffs["d"])

    return result_list


def pretty_solve_cubic_splines_table(table: pd.DataFrame) -> dict[str, Any]:

    # Берем строку, преобразовываем ее в список, берем все элементы списка после первого (первый - название строки)
    x: np.ndarray = np.array(table.iloc[0].tolist()[1:], dtype=float)
    y: np.ndarray = np.array(table.iloc[1].tolist()[1:], dtype=float)

    input_rows: list[dict[str, Any]] = []
    h_delta_rows: list[dict[str, Any]] = []
    system_rows: list[dict[str, Any]] = []
    coeff_rows: list[dict[str, Any]] = []

    spline_rows: list[dict[str, Any]] = []

    boundary_rows: list[dict[str, Any]] = []

    for i in range(len(x)):
        input_rows.append(
            {
                "i": i,
                "x_i": x[i],
                "y_i": y[i] if i < len(y) else None,
            }
        )

    if len(np.unique(x)) != len(x):
        return {
            "status": "error",
            "message": "Узлы x не должны повторяться",
            "x": x.copy(),
            "y": y.copy(),
            "input_table": pd.DataFrame(input_rows),
        }

    if len(x) != len(y):
        return {
            "status": "error",
            "message": "Количество x и y не совпадает",
            "x": x.copy(),
            "y": y.copy(),
            "input_table": pd.DataFrame(input_rows),
        }

    if len(x) < 2 or len(y) < 2:
        return {
            "status": "error",
            "message": "Не хватает значений для построения сплайна. Нужно >= 2 значений",
            "x": x.copy(),
            "y": y.copy(),
            "input_table": pd.DataFrame(input_rows),
        }

    coeffs: dict = {"a": [], "b": [], "c": [], "d": []}
    n: int = len(x)
    nodes_count: int = n - 1

    h: np.ndarray = np.zeros(nodes_count, dtype=float)
    deltas: np.ndarray = np.zeros(nodes_count, dtype=float)
    a: np.ndarray = np.zeros(nodes_count, dtype=float)
    b: np.ndarray = np.zeros(nodes_count, dtype=float)

    c: np.ndarray = np.zeros(n, dtype=float)

    d: np.ndarray = np.zeros(nodes_count, dtype=float)

    for i in range(nodes_count):
        h[i] = x[i + 1] - x[i]

        if h[i] == 0:
            return {
                "status": "error",
                "message": "Два соседних узла x совпадают, h_i = 0",
                "x": x.copy(),
                "y": y.copy(),
                "input_table": pd.DataFrame(input_rows),
                "h": h.copy(),
                "deltas": deltas.copy(),
                "h_delta_table": pd.DataFrame(h_delta_rows),
            }

        deltas[i] = (y[i + 1] - y[i]) / h[i]

        h_delta_rows.append(
            {
                "i": i,
                "x_i": x[i],
                "x_i+1": x[i + 1],
                "y_i": y[i],
                "y_i+1": y[i + 1],
                "h_i = x_i+1 - x_i": h[i],
                "delta_i = (y_i+1 - y_i) / h_i": deltas[i],
            }
        )

    A_matrix: np.ndarray = np.zeros((nodes_count - 1, nodes_count - 1), dtype=float)
    b_vec: np.ndarray = np.zeros(nodes_count - 1, dtype=float)

    for i in range(1, n - 1):
        row: int = i - 1
        a_tmp: float = h[i - 1]
        b_tmp: float = 2 * (h[i - 1] + h[i])
        c_tmp: float = h[i]

        A_matrix[row] = matrix_diag.build_diag_row(
            a_tmp, b_tmp, c_tmp, A_matrix.shape[0], row
        )

        b_vec[row] = 3 * (deltas[i] - deltas[i - 1])

        system_rows.append(
            {
                "i": i,
                "row": row,
                "lower_diag = h_i-1": a_tmp,
                "main_diag = 2(h_i-1 + h_i)": b_tmp,
                "upper_diag = h_i": c_tmp,
                "right_part = 3(delta_i - delta_i-1)": b_vec[row],
            }
        )

    if nodes_count == 1:
        res = (np.array([], dtype=float), np.array([], dtype=float))
    else:
        res = clean_solve_progonka_matrix(A_matrix, b_vec)
    if res is None:
        return {
            "status": "error",
            "message": "Система для коэффициентов c не решена методом Гаусса",
            "x": x.copy(),
            "y": y.copy(),
            "h": h.copy(),
            "deltas": deltas.copy(),
            "A_matrix": A_matrix.copy(),
            "b_vec": b_vec.copy(),
            "input_table": pd.DataFrame(input_rows),
            "h_delta_table": pd.DataFrame(h_delta_rows),
            "system_table": pd.DataFrame(system_rows),
        }

    c[1:nodes_count] = np.array(res[0], dtype=float)

    h_func_res: tuple | None = get_h_and_deltas(x, y)
    if h_func_res is None:
        return {
            "status": "error",
            "message": "Ошибка при повторном вычислении h и delta",
            "x": x.copy(),
            "y": y.copy(),
            "h": h.copy(),
            "deltas": deltas.copy(),
            "A_matrix": A_matrix.copy(),
            "b_vec": b_vec.copy(),
            "c": c.copy(),
            "gauss_result": res,
            "input_table": pd.DataFrame(input_rows),
            "h_delta_table": pd.DataFrame(h_delta_rows),
            "system_table": pd.DataFrame(system_rows),
        }

    h, deltas = h_func_res

    for i in range(nodes_count):
        a[i] = y[i]

        d[i] = (c[i + 1] - c[i]) / (3 * h[i])
        b[i] = deltas[i] - h[i] / 3 * (2 * c[i] + c[i + 1])

        coeff_rows.append(
            {
                "i": i,
                "interval": f"[{x[i]}, {x[i + 1]}]",
                "a_i": a[i],
                "b_i": b[i],
                "c_i": c[i],
                "c_i+1": c[i + 1],
                "d_i": d[i],
            }
        )

        spline_rows.append(
            {
                "i": i,
                "interval": f"[{x[i]}, {x[i + 1]}]",
                "formula": (
                    f"S_{i}(x) = "
                    f"{a[i]} + ({b[i]})(x - {x[i]}) + "
                    f"({c[i]})(x - {x[i]})^2 + "
                    f"({d[i]})(x - {x[i]})^3"
                ),
                "second_derivative": (
                    f"S_{i}''(x) = " f"{2 * c[i]} + ({6 * d[i]})(x - {x[i]})"
                ),
            }
        )

    coeffs["a"] = a
    coeffs["b"] = b
    coeffs["c"] = c
    coeffs["d"] = d

    coeff_matrix: np.ndarray = np.column_stack((a, b, c[:-1], d))

    s2_left: float = 2 * c[0]
    s2_right: float = 2 * c[-1]
    s2_diff: float = s2_left - s2_right

    boundary_rows.append(
        {
            "condition": "s''(x_1) = s''(x_6)",
            "s''(x_1)": s2_left,
            "s''(x_6)": s2_right,
            "difference": s2_diff,
            "is_satisfied": bool(np.isclose(s2_left, s2_right)),
            "note": (
                "Условие выполнено как частный случай натурального сплайна: "
                "c[0] = 0, c[-1] = 0, поэтому s''(x_1) = s''(x_6) = 0"
            ),
        }
    )

    return {
        "status": "ok",
        "coefficients": coeff_matrix.copy(),
        "a": a.copy(),
        "b": b.copy(),
        "c": c.copy(),
        "d": d.copy(),
        "x": x.copy(),
        "y": y.copy(),
        "h": h.copy(),
        "deltas": deltas.copy(),
        "A_matrix": A_matrix.copy(),
        "b_vec": b_vec.copy(),
        "gauss_result": res,
        "input_table": pd.DataFrame(input_rows),
        "h_delta_table": pd.DataFrame(h_delta_rows),
        "system_table": pd.DataFrame(system_rows),
        "coefficients_table": pd.DataFrame(coeff_rows),
        "spline_table": pd.DataFrame(spline_rows),
        "boundary_condition_table": pd.DataFrame(boundary_rows),
        "s2_left": s2_left,
        "s2_right": s2_right,
        "s2_diff": s2_diff,
        "boundary_condition_satisfied": bool(np.isclose(s2_left, s2_right)),
        "boundary_condition_note": (
            "Использован натуральный сплайн: c[0] = 0, c[-1] = 0. "
            "Поэтому s''(x_1) = s''(x_6)."
        ),
        "nodes_count": nodes_count,
        "n": n,
    }
