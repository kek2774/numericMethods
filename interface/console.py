from __future__ import annotations

import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

CONSOLE_WIDTH = 118
console = Console(width=CONSOLE_WIDTH, color_system=None, highlight=False)

COMPACT_TABLES = True
COMPACT_TABLE_LIMIT = 8
COMPACT_TABLE_EDGE = 3


def configure_output(*, compact_tables: bool = True) -> None:
    global COMPACT_TABLES
    COMPACT_TABLES = compact_tables


LABELS: dict[str, str] = {
    "status": "Статус",
    "message": "Сообщение",
    "solution": "Решение",
    "solutions": "Решение",
    "result": "Результат",
    "residual": "Невязка",
    "residuals": "Невязки",
    "residual_norm_inf": "||r||∞",
    "residuals_norm_inf": "||r||∞",
    "matrix_after_forward_pass": "Матрица после прямого хода",
    "transition_matrix": "Матрица перехода",
    "transition_vector": "Вектор перехода",
    "transition_matrix_norms": "Нормы матрицы перехода",
    "chosen_norm": "Выбранная норма",
    "chosen_norm_name": "Название нормы",
    "iterations": "Итераций",
    "A_coeff": "Коэффициенты A_i",
    "B_coeff": "Коэффициенты B_i",
    "T_matrix": "Матрица T",
    "d_vec": "Вектор d",
    "coefficients": "Коэффициенты",
    "divided_differences_matrix": "Разделенные разности",
    "a": "a",
    "b": "b",
    "c": "c",
    "h": "h",
    "x": "x",
    "y": "y",
    "x_0": "x_0",
    "y_0": "y_0",
    "x_star": "x*",
    "s2_left": "S''(x_1)",
    "s2_right": "S''(x_6)",
    "s2_diff": "Разность S''",
    "uniform_grid": "Равномерная сетка",
    "q": "q",
    "lambda_val": "λ",
    "eps": "ε",
    "n": "n",
    "iteration_table": "Таблица итераций",
    "coefficient_table": "Коэффициенты",
    "coefficients_table": "Коэффициенты",
    "forward_pass_table": "Прямой ход",
    "backward_pass_table": "Обратный ход",
    "residual_table": "Невязки",
    "terms_table": "Слагаемые",
    "factors_table": "Множители",
    "h_delta_table": "Шаги и разности",
    "system_table": "Система для коэффициентов",
    "spline_table": "Сплайны",
    "boundary_condition_table": "Граничное условие",
    "grid_table": "Сетка",
    "boundary_table": "Краевые условия",
    "equation_table": "Уравнения в узлах",
    "step_table": "Шаги метода",
    "interpolation_table": "Интерполяция",
    "diff_table": "Разностная таблица",
    "input_table": "Исходная таблица",
}


COLUMN_LABELS: dict[str, str] = {
    "iteration": "k",
    "iterations": "k",
    "accuracy": "оценка ε",
    "x_diff_norm": "||x_k-x_{k-1}||",
    "error_estimate": "оценка ε",
    "solution": "решение",
    "solutions": "решение",
    "residual": "невязка",
    "residuals": "невязки",
    "residuals_norm_inf": "||r||∞",
    "residual_norm_inf": "||r||∞",
    "condition": "условие",
    "is_satisfied": "выполнено",
    "message": "сообщение",
    "method": "метод",
    "side": "край",
    "value": "значение",
    "result": "результат",
    "nodes": "узлы",
    "point": "точка",
    "formula": "формула",
    "interval": "отрезок",
    "product": "произведение",
    "term": "слагаемое",
    "partial_sum": "частичная сумма",
    "denominator": "знаменатель",
    "right_part": "правая часть",
    "right_part = 3(delta_i - delta_i-1)": "3(δ_i-δ_{i-1})",
    "second_derivative": "вторая производная",
    "lower_diag = h_i-1": "нижняя диаг.",
    "main_diag = 2(h_i-1 + h_i)": "главная диаг.",
    "upper_diag = h_i": "верхняя диаг.",
    "h_i = x_i+1 - x_i": "h_i",
    "delta_i = (y_i+1 - y_i) / h_i": "δ_i",
    "r_i = b_i - Ax_i": "r_i",
    "||x_n - x_n-1||": "||x_k-x_{k-1}||",
    "|x_n+1 - x_n|": "|x_{k+1}-x_k|",
}


def section(title: str) -> None:
    console.print()
    console.print(Rule(title, characters="=", style="bold"))


def subsection(title: str) -> None:
    console.print()
    console.print(Rule(title, characters="-"))


def line(label: str, value: Any) -> None:
    console.print(Text.assemble((f"{label}: ", "bold"), str(value)))


def status_text(status: Any) -> str:
    if status == "ok":
        return "успешно"
    if status == "error":
        return "ошибка"
    return str(status)


def clean_float(value: float) -> float:
    value = float(value)
    if abs(value) < 0.5e-12:
        return 0.0
    return value


def format_number(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{clean_float(value):.6f}"


INTEGER_COLUMNS = {
    "i",
    "k",
    "n",
    "n_prev",
    "row",
    "j",
}


def format_cell(value: Any, column: str | None = None) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "да" if value else "нет"
    if (
        column in INTEGER_COLUMNS
        and isinstance(value, (float, np.floating, int, np.integer))
        and not pd.isna(value)
    ):
        return str(int(round(float(value))))
    if isinstance(value, (float, np.floating)):
        return format_number(float(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if pd.isna(value):
        return "-"
    return str(value)


def format_array(value: Any) -> str:
    array = np.asarray(value)

    def item_to_text(item: Any) -> str:
        if isinstance(item, (float, np.floating)) and np.isnan(item):
            return "·"
        if isinstance(item, (float, np.floating, int, np.integer)):
            return f"{clean_float(float(item)):.6f}"
        return str(item)

    if array.ndim == 2:
        rows = [[item_to_text(item) for item in row] for row in array]
        widths = [
            max(len(rows[row_index][col_index]) for row_index in range(len(rows)))
            for col_index in range(array.shape[1])
        ]
        return "\n".join(
            "["
            + " ".join(cell.rjust(widths[index]) for index, cell in enumerate(row))
            + "]"
            for row in rows
        )

    return np.array2string(
        array,
        formatter={"float_kind": item_to_text},
        separator=" ",
        max_line_width=CONSOLE_WIDTH,
    )


def print_task_brief(
    task: str,
    condition: str,
    restrictions: Iterable[str],
    answers: Iterable[str],
) -> None:
    rows = Table.grid(expand=True, padding=(0, 1))
    rows.add_column("label", no_wrap=True, width=16)
    rows.add_column("value", overflow="fold")
    rows.add_row(Text("Задача", style="bold"), Text(task))
    rows.add_row(Text("Условие", style="bold"), Text(condition))

    restriction_lines = [Text("Ограничения / исходные данные", style="bold")]
    for item in restrictions:
        restriction_lines.append(Text(f"- {item}"))

    answer_lines = [Text("Ответ", style="bold green")]
    for item in answers:
        answer_lines.append(Text(f"- {item}"))

    console.print(
        Panel(
            Group(rows, Text(), *restriction_lines, Text(), *answer_lines),
            title="Основные сведения",
            box=box.ROUNDED,
            expand=True,
        )
    )


def print_text_block(title: str, lines: Iterable[str]) -> None:
    body = "\n".join(lines)
    console.print(Panel(Text(body), title=title, box=box.ROUNDED, expand=True))


def print_value(label: str, value: Any) -> None:
    label = LABELS.get(label, label)
    if isinstance(value, pd.DataFrame):
        print_table(label, value)
        return

    if isinstance(value, np.ndarray):
        console.print(Panel(format_array(value), title=label, box=box.SIMPLE, expand=False))
        return

    if isinstance(value, (list, tuple)) and value and all(
        isinstance(item, np.ndarray) for item in value
    ):
        for index, item in enumerate(value, start=1):
            console.print(
                Panel(
                    format_array(item),
                    title=f"{label} [{index}]",
                    box=box.SIMPLE,
                    expand=False,
                )
            )
        return

    line(label, format_cell(value))


def localized_columns(table: pd.DataFrame) -> pd.DataFrame:
    return table.rename(columns={col: COLUMN_LABELS.get(str(col), str(col)) for col in table.columns})


def compact_dataframe(table: pd.DataFrame) -> pd.DataFrame:
    if not COMPACT_TABLES or len(table) <= COMPACT_TABLE_LIMIT:
        return table

    marker = {column: "…" for column in table.columns}
    return pd.concat(
        [
            table.head(COMPACT_TABLE_EDGE),
            pd.DataFrame([marker]),
            table.tail(COMPACT_TABLE_EDGE),
        ],
        ignore_index=True,
    )


def print_table(title: str, table: pd.DataFrame) -> None:
    title = LABELS.get(title, title)
    if table.empty:
        console.print(Panel("(пусто)", title=title, box=box.SIMPLE, expand=False))
        return

    source = table.copy()
    if title != "Сводка результатов":
        source = compact_dataframe(source)
    prepared = localized_columns(source)
    rich_table = Table(
        title=title if title else None,
        box=box.SIMPLE_HEAVY,
        show_lines=False,
        expand=False,
        safe_box=True,
    )

    for column in prepared.columns:
        rich_table.add_column(str(column), justify="right", overflow="fold", no_wrap=False)

    for _, row in prepared.iterrows():
        rich_table.add_row(
            *(format_cell(value, str(column)) for column, value in row.items())
        )

    console.print(rich_table)


def print_result(
    title: str,
    result: dict[str, Any],
    values: Iterable[str] = (),
    tables: Iterable[str] = (),
) -> None:
    subsection(title)
    line("Статус", status_text(result.get("status", "unknown")))
    if result.get("message"):
        line("Сообщение", result["message"])

    for key in values:
        if key in result:
            print_value(key, result[key])

    for key in tables:
        value = result.get(key)
        if isinstance(value, pd.DataFrame):
            print_table(key, value)
