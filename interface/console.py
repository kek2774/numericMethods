from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text


CONSOLE_WIDTH = 118
console = Console(width=CONSOLE_WIDTH, color_system=None, highlight=False)


LABELS: dict[str, str] = {
    "status": "Статус",
    "message": "Сообщение",
    "solution": "Решение",
    "solutions": "Решение",
    "result": "Результат",
    "residual": "Невязка",
    "residuals": "Невязки",
    "residual_norm_inf": "||r||inf",
    "residuals_norm_inf": "||r||inf",
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
    "x_0": "x0",
    "y_0": "y0",
    "x_star": "x*",
    "s2_left": "S''(x1)",
    "s2_right": "S''(x6)",
    "s2_diff": "Разность S''",
    "uniform_grid": "Равномерная сетка",
    "q": "q",
    "lambda_val": "lambda",
    "eps": "eps",
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
    "accuracy": "оценка",
    "x_diff_norm": "||x(k)-x(k-1)||",
    "error_estimate": "оценка",
    "solution": "решение",
    "solutions": "решение",
    "residual": "невязка",
    "residuals": "невязки",
    "residuals_norm_inf": "||r||inf",
    "residual_norm_inf": "||r||inf",
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
    "lower_diag = h_i-1": "нижняя диаг.",
    "main_diag = 2(h_i-1 + h_i)": "главная диаг.",
    "upper_diag = h_i": "верхняя диаг.",
    "h_i = x_i+1 - x_i": "h_i",
    "delta_i = (y_i+1 - y_i) / h_i": "delta_i",
    "r_i = b_i - Ax_i": "r_i",
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


def format_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, (float, np.floating)):
        return format_number(float(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if pd.isna(value):
        return "-"
    return str(value)


def format_array(value: Any) -> str:
    array = np.asarray(value)

    def float_formatter(item: float) -> str:
        if np.isnan(item):
            return "-"
        return f"{clean_float(item):.6f}"

    return np.array2string(
        array,
        formatter={"float_kind": float_formatter},
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
    rows.add_column("label", style="bold", no_wrap=True, width=16)
    rows.add_column("value", overflow="fold")
    rows.add_row("Задача", task)
    rows.add_row("Условие", condition)

    restriction_lines = [Text("Ограничения / заданные данные", style="bold")]
    for item in restrictions:
        restriction_lines.append(Text(f"- {item}"))

    answer_lines = [Text("Ответ", style="bold green")]
    for item in answers:
        answer_lines.append(Text(f"- {item}"))

    console.print(
        Panel(
            Group(rows, Text(), *restriction_lines, Text(), *answer_lines),
            title="Задача -> ответ",
            box=box.ROUNDED,
            expand=True,
        )
    )


def print_text_block(title: str, lines: Iterable[str]) -> None:
    body = "\n".join(lines)
    console.print(Panel(body, title=title, box=box.ROUNDED, expand=True))


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


def print_table(title: str, table: pd.DataFrame) -> None:
    title = LABELS.get(title, title)
    if table.empty:
        console.print(Panel("(пусто)", title=title, box=box.SIMPLE, expand=False))
        return

    prepared = localized_columns(table.copy())
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
        rich_table.add_row(*(format_cell(value) for value in row.tolist()))

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
