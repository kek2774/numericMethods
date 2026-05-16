"""Flet интерфейс для лабораторной работы по численным методам."""

from __future__ import annotations

import traceback
from dataclasses import dataclass, replace
from typing import Any, Callable

import flet as ft
import flet_charts as fch
import flet_datatable2 as fdt
import numpy as np
import pandas as pd
import sympy as sp

from methods.adams_func import pretty_solve_adams_func
from methods.cube_spline_table import pretty_solve_cubic_splines_table
from methods.dihotomy_func import pretty_solve_dihotomy_func
from methods.first_diff_table import pretty_solve_first_diff_table
from methods.gauss_matrix import pretty_solve_gauss_matrix
from methods.lagrange_table import pretty_solve_lagrange_table
from methods.mls_linear_table import pretty_solve_linear_mls_table
from methods.mls_quadratic_table import pretty_solve_quadratic_mls_table
from methods.mpi_func import pretty_solve_mpi_matrix as pretty_solve_mpi_func
from methods.mpi_matrix import pretty_solve_mpi_matrix as pretty_solve_mpi_slay
from methods.newton_func import pretty_solve_newton_func
from methods.newton_table import pretty_solve_newton_table
from methods.progonka_func import pretty_solve_progonka_func
from methods.progonka_matrix import pretty_solve_progonka_matrix
from methods.runge_cutta_func import pretty_solve_runge_cutta_func
from methods.second_diff_table import pretty_solve_second_diff_table
from methods.simpson_integration_func import pretty_solve_simpson_integration_func
from methods.zeidel_matrix import pretty_solve_zeidel_matrix


LABELS: dict[str, str] = {
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
    "iterations": "Итераций",
    "A_coeff": "Коэффициенты A_i",
    "B_coeff": "Коэффициенты B_i",
    "T_matrix": "Матрица T",
    "d_vec": "Вектор d",
    "coefficients": "Коэффициенты",
    "divided_differences_matrix": "Разделённые разности",
    "a": "a", "b": "b", "c": "c", "h": "h",
    "x": "x", "y": "y", "x_0": "x₀", "y_0": "y₀", "x_star": "x*",
    "s2_left": "S''(x₁)", "s2_right": "S''(x₆)", "s2_diff": "Разность S''",
    "uniform_grid": "Равномерная сетка",
    "q": "q", "lambda_val": "λ",
    "eps": "eps", "n": "n",
    "iteration_table": "Таблица итераций",
    "coefficient_table": "Коэффициенты",
    "coefficients_table": "Коэффициенты",
    "forward_pass_table": "Прямой ход",
    "backward_pass_table": "Обратный ход",
    "residual_table": "Невязки",
    "terms_table": "Слагаемые",
    "h_delta_table": "Шаги и разности",
    "system_table": "Система для коэффициентов",
    "spline_table": "Сплайны",
    "grid_table": "Сетка",
    "boundary_table": "Краевые условия",
    "equation_table": "Уравнения в узлах",
    "answer_table": "Ответ в заданных точках",
    "solution_table": "Значения решения",
    "step_table": "Шаги метода",
    "interpolation_table": "Интерполяция",
    "diff_table": "Разностная таблица",
}


PRIMARY = ft.Colors.INDIGO_500
PRIMARY_DARK = ft.Colors.INDIGO_700
ACCENT = ft.Colors.RED_500
SECONDARY = ft.Colors.ORANGE_500
TERTIARY = ft.Colors.GREEN_600
NEUTRAL = ft.Colors.BLUE_GREY_400


# ===== Парсер выражений на sympy =====

_SYMPY_LOCALS = {
    "lg": lambda v: sp.log(v, 10),
    "ln": sp.log,
}


def make_function_1arg(expr_str: str) -> Callable[[float], float]:
    x = sp.Symbol("x")
    expr = sp.sympify(expr_str, locals=_SYMPY_LOCALS)
    raw = sp.lambdify(x, expr, modules=["math"])
    return lambda x_val: float(raw(x_val))


def make_function_2arg(expr_str: str) -> Callable[[float, float], float]:
    x, y = sp.symbols("x y")
    expr = sp.sympify(expr_str, locals=_SYMPY_LOCALS)
    raw = sp.lambdify((x, y), expr, modules=["math"])
    return lambda x_val, y_val: float(raw(x_val, y_val))


# ===== Форматирование =====

_SUBSCRIPT_DIGITS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def sub(n: int) -> str:
    return str(n).translate(_SUBSCRIPT_DIGITS)


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if abs(v) < 0.5e-12:
            v = 0.0
        return f"{v:.6f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def fmt_array(arr: Any) -> str:
    arr = np.asarray(arr)
    return np.array2string(
        arr,
        precision=6,
        suppress_small=True,
        separator=", ",
        floatmode="fixed",
        max_line_width=200,
    )


# ===== LaTeX через ft.Markdown =====

LATEX_TEXT_COLOR = "#0F172A"
LATEX_FONT_FAMILY = "Times New Roman"
LATEX_FONT_FALLBACK = ["Cambria Math", "Cambria", "Georgia", "serif"]
LATEX_BASE = 14         # базовый размер LaTeX-шрифта
LATEX_SIZE_LARGE = 22   # ответы
LATEX_SIZE_MEDIUM = 18  # обычные выводы
LATEX_SIZE_SMALL = 17   # заголовки таблиц / формулы-многострочники
LATEX_SIZE_TINY = 15    # ячейки таблиц / легенда


def _latex_text_style() -> ft.TextStyle:
    return ft.TextStyle(
        color=LATEX_TEXT_COLOR,
        bgcolor=ft.Colors.TRANSPARENT,
        font_family=LATEX_FONT_FAMILY,
        font_family_fallback=LATEX_FONT_FALLBACK,
    )


def latex(expr: str, fontsize: int = LATEX_SIZE_MEDIUM) -> ft.Control:
    """LaTeX-выражение через нативный рендер ft.Markdown ($$...$$).

    Используем только latex_scale_factor для масштаба — без size в style,
    чтобы избежать двойного скейла.
    """
    return ft.Markdown(
        value=f"$${expr}$$",
        extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
        latex_scale_factor=fontsize / LATEX_BASE,
        latex_style=_latex_text_style(),
        selectable=True,
        fit_content=True,
    )


def latex_inline(expr: str, fontsize: int = LATEX_SIZE_TINY) -> ft.Control:
    """Inline-LaTeX для лейблов/ячеек."""
    return ft.Markdown(
        value=f"${expr}$",
        extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
        latex_scale_factor=fontsize / LATEX_BASE,
        latex_style=_latex_text_style(),
        selectable=True,
        fit_content=True,
    )


def latex_num(value: Any, precision: int = 6) -> str:
    """Число в LaTeX-формате."""
    if value is None:
        return r"\text{—}"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(v) < 0.5e-12:
        v = 0.0
    if v != 0 and (abs(v) < 1e-3 or abs(v) > 1e5):
        s = f"{v:.{precision - 1}e}"
        mantissa, exp = s.split("e")
        return rf"{mantissa} \cdot 10^{{{int(exp)}}}"
    return f"{v:.{precision}f}".rstrip("0").rstrip(".") or "0"


def latex_signed_term(value: float, factor: str = "") -> str:
    if abs(value) < 0.5e-12:
        return ""
    sign = "+" if value > 0 else "-"
    body = latex_num(abs(value))
    if factor:
        return rf" {sign} {body}\,{factor}"
    return rf" {sign} {body}"


def latex_braced_system(lines: list[str], fontsize: int = LATEX_SIZE_LARGE) -> ft.Control:
    if not lines:
        return ft.Container()
    body = r" \\ ".join(lines)
    expr = rf"\begin{{cases}} {body} \end{{cases}}"
    return latex(expr, fontsize=fontsize)


def latex_matrix(matrix: np.ndarray, fontsize: int = LATEX_SIZE_MEDIUM) -> ft.Control:
    arr = np.asarray(matrix)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.size == 0:
        return ft.Container()
    rows = r" \\ ".join(" & ".join(latex_num(v) for v in row) for row in arr)
    expr = rf"\begin{{pmatrix}} {rows} \end{{pmatrix}}"
    return latex(expr, fontsize=fontsize)


def latex_augmented_matrix(A: np.ndarray, b: np.ndarray, fontsize: int = LATEX_SIZE_MEDIUM) -> ft.Control:
    A_arr = np.asarray(A, dtype=float)
    b_arr = np.asarray(b, dtype=float).reshape(-1, 1)
    if A_arr.ndim != 2 or b_arr.shape[0] != A_arr.shape[0]:
        return latex_matrix(A_arr, fontsize=fontsize)
    columns = "c" * A_arr.shape[1] + "|c"
    rows = []
    for a_row, b_value in zip(A_arr, b_arr.flatten()):
        left = " & ".join(latex_num(v) for v in a_row)
        rows.append(f"{left} & {latex_num(b_value)}")
    body = r" \\ ".join(rows)
    return latex(rf"\left(\begin{{array}}{{{columns}}} {body} \end{{array}}\right)", fontsize=fontsize)


def latex_vector(arr: np.ndarray, fontsize: int = LATEX_SIZE_MEDIUM, column: bool = True) -> ft.Control:
    """По умолчанию — столбец."""
    flat = np.asarray(arr).flatten()
    sep = r" \\ " if column else r" & "
    body = sep.join(latex_num(v) for v in flat)
    expr = rf"\begin{{pmatrix}} {body} \end{{pmatrix}}"
    return latex(expr, fontsize=fontsize)


# ===== Общие виджеты =====

def card(title: str, *content: ft.Control) -> ft.Container:
    return ft.Container(
        content=ft.Column(
            controls=[
                ft.Text(title, size=16, weight=ft.FontWeight.W_600),
                ft.Divider(height=1),
                *content,
            ],
            spacing=10,
            tight=True,
        ),
        padding=ft.Padding.all(16),
        bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
        border_radius=12,
    )


def section_title(text: str, subtitle: str = "") -> ft.Column:
    items: list[ft.Control] = [ft.Text(text, size=24, weight=ft.FontWeight.BOLD)]
    if subtitle:
        items.append(ft.Text(subtitle, size=13, color=ft.Colors.ON_SURFACE_VARIANT))
    return ft.Column(controls=items, spacing=4)


LATEX_COLUMN_NAMES: dict[str, str] = {
    "iteration": "k",
    "iterations": "k",
    "x_n": "x_n",
    "x_n+1": "x_{n+1}",
    "x_i": "x_i",
    "x_i-1": "x_{i-1}",
    "x_i+1": "x_{i+1}",
    "y_i": "y_i",
    "y_i-1": "y_{i-1}",
    "y_i+1": "y_{i+1}",
    "f_i": "f_i",
    "f_i-1": "f_{i-1}",
    "|x_n+1 - x_n|": r"|x_{n+1} - x_n|",
    "accuracy": r"\text{оценка}",
    "x_diff_norm": r"\|\Delta x\|",
    "error_estimate": r"\text{оценка}",
    "x_0": "x_0",
    "f(x_0)": "f(x_0)",
    "a": "a", "b": "b",
    "f(a)": "f(a)", "f(b)": "f(b)",
    "|a - b|": "|a - b|",
    "f(x_n)": "f(x_n)",
    "f'(x_n)": "f'(x_n)",
    "solution": r"x",
    "solutions": r"x",
    "residual": "r",
    "residuals": "r",
    "residuals_norm_inf": r"\|r\|_{\infty}",
    "residual_norm_inf": r"\|r\|_{\infty}",
    "x": "x", "y": "y",
    "y'": "y'",
    "y''": "y''",
    "y'_i": "y'_i",
    "y''_i": "y''_i",
    "n": "n",
    "S": "S",
    "n_prev": r"n_{\mathrm{prev}}",
    "S_prev": r"S_{\mathrm{prev}}",
    "|S - S_prev| / 15": r"\frac{|S - S_{\mathrm{prev}}|}{15}",
    "k1": "k_1",
    "k2": "k_2",
    "k3": "k_3",
    "k4": "k_4",
    "h_i": "h_i", "delta_i": r"\delta_i",
    "h_i = x_i+1 - x_i": "h_i",
    "delta_i = (y_i+1 - y_i) / h_i": r"\delta_i",
    "lower_diag = h_i-1": r"h_{i-1}",
    "main_diag = 2(h_i-1 + h_i)": r"2(h_{i-1} + h_i)",
    "upper_diag = h_i": "h_i",
    "right_part": r"\text{правая часть}",
    "right_part = 3(delta_i - delta_i-1)": r"3(\delta_i - \delta_{i-1})",
    "row": r"\text{строка}",
    "a_i": "a_i",
    "b_i": "b_i",
    "c_i": "c_i",
    "c_i+1": "c_{i+1}",
    "d_i": "d_i",
    "second_derivative": r"S_i''(x)",
    "s''(x_1)": r"S''(x_1)",
    "s''(x_6)": r"S''(x_n)",
    "difference": r"\Delta",
    "note": r"\text{примечание}",
    "r_i = b_i - Ax_i": "r_i",
    "i": "i", "j": "j",
    "term": r"\text{слагаемое}",
    "partial_sum": r"\text{част. сумма}",
    "denominator": r"\text{знам.}",
    "product": r"\text{произв.}",
    "nodes": r"\text{узлы}",
    "point": r"\text{точка}",
    "formula": r"\text{формула}",
    "interval": r"\text{отрезок}",
    "method": r"\text{метод}",
    "condition": r"\text{условие}",
    "is_satisfied": r"\text{выполн.}",
    "message": r"\text{сообщ.}",
    "value": r"\text{значение}",
    "result": r"\text{результат}",
    "side": r"\text{край}",
}


def _column_label_latex(name: str) -> str:
    """LaTeX-представление имени столбца таблицы."""
    if name in LATEX_COLUMN_NAMES:
        return LATEX_COLUMN_NAMES[name]
    if "_" in name and not any(ch in name for ch in "^+-*/|()"):
        return rf"\text{{{name.replace('_', ' ')}}}"
    # Если имя похоже на формулу (содержит спецсимволы), используем как есть
    if any(ch in name for ch in "_^+-*/|()"):
        return name
    return rf"\text{{{name}}}"


def _looks_like_math_cell(value: str) -> bool:
    return "=" in value and any(marker in value for marker in ("_", "^", "''", "\\"))


def df_to_datatable(df: pd.DataFrame) -> ft.Control:
    if df.empty:
        return ft.Text("(пусто)", italic=True, color=ft.Colors.ON_SURFACE_VARIANT)
    column_names = [str(c) for c in df.columns]

    def make_columns() -> list[fdt.DataColumn2]:
        return [
            fdt.DataColumn2(
                label=latex_inline(_column_label_latex(name), fontsize=LATEX_SIZE_SMALL),
                fixed_width=74 if index == 0 else None,
                size=fdt.DataColumnSize.S,
            )
            for index, name in enumerate(column_names)
        ]

    rows: list[fdt.DataRow2] = []
    for _, row in df.iterrows():
        cells: list[ft.DataCell] = []
        for value in row.tolist():
            if isinstance(value, bool):
                cell_ctrl = ft.Text("да" if value else "нет", size=12, color=LATEX_TEXT_COLOR)
            elif isinstance(value, (float, np.floating, int, np.integer)):
                cell_ctrl = latex_inline(latex_num(value), fontsize=LATEX_SIZE_TINY)
            elif value is None or (isinstance(value, float) and pd.isna(value)):
                cell_ctrl = ft.Text("-", size=12, color=LATEX_TEXT_COLOR)
            else:
                text_value = str(value)
                if _looks_like_math_cell(text_value):
                    cell_ctrl = latex_inline(text_value, fontsize=LATEX_SIZE_TINY)
                else:
                    cell_ctrl = ft.Text(text_value, size=12, color=LATEX_TEXT_COLOR)
            cells.append(ft.DataCell(cell_ctrl))
        rows.append(fdt.DataRow2(cells=cells, specific_row_height=56))

    row_height = 56
    heading_height = 58
    visible_rows = min(max(len(rows), 1), 8)
    table_height = heading_height + visible_rows * row_height + 18
    table = fdt.DataTable2(
        columns=make_columns(),
        rows=rows,
        fixed_top_rows=1,
        fixed_left_columns=1 if len(column_names) > 1 else 0,
        fixed_columns_color=ft.Colors.SURFACE_CONTAINER_HIGHEST,
        fixed_corner_color=ft.Colors.SURFACE_CONTAINER_HIGHEST,
        heading_row_color=ft.Colors.SURFACE_CONTAINER_HIGH,
        heading_row_height=heading_height,
        data_row_height=row_height,
        horizontal_margin=12,
        column_spacing=20,
        min_width=max(720, 92 + 138 * max(len(column_names) - 1, 1)),
        height=table_height,
        show_bottom_border=True,
        horizontal_lines=ft.BorderSide(0.6, ft.Colors.OUTLINE_VARIANT),
        visible_horizontal_scroll_bar=True,
        visible_vertical_scroll_bar=len(rows) > visible_rows,
    )
    return ft.Container(
        content=table,
        padding=ft.Padding.symmetric(vertical=4),
    )


LATEX_KEY_LHS: dict[str, str] = {
    "residual_norm_inf": r"\|r\|_{\infty}",
    "residuals_norm_inf": r"\|r\|_{\infty}",
    "s2_left": r"S''(x_1)",
    "s2_right": r"S''(x_n)",
    "s2_diff": r"\Delta S''",
    "lambda_val": r"\lambda",
    "q": "q",
    "eps": r"\varepsilon",
    "h": "h",
    "iterations": r"\text{итераций}",
    "x_0": "x_0",
    "y_0": "y_0",
    "x_star": "x^{*}",
    "result": r"\text{результат}",
    "n": "n",
    "a": "a", "b": "b", "c": "c",
}


def _latex_scalar_block(label: str, value: float) -> ft.Control:
    lhs = LATEX_KEY_LHS.get(label, LABELS.get(label, label))
    if label not in LATEX_KEY_LHS:
        # для лейблов, которых нет в LATEX_KEY_LHS — оборачиваем в \text{}
        lhs = rf"\text{{{lhs}}}"
    expr = f"{lhs} = {latex_num(value)}"
    return latex(expr, fontsize=LATEX_SIZE_MEDIUM)


def value_block(label: str, value: Any) -> ft.Control:
    title = LABELS.get(label, label)
    if isinstance(value, pd.DataFrame):
        return ft.Column([ft.Text(title, weight=ft.FontWeight.W_600), df_to_datatable(value)], spacing=6)
    if (
        label == "matrix_after_forward_pass"
        and isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(v, np.ndarray) for v in value)
    ):
        return ft.Column(
            [
                ft.Text(title, weight=ft.FontWeight.W_600),
                ft.Container(
                    content=latex_augmented_matrix(value[0], value[1], fontsize=LATEX_SIZE_MEDIUM),
                    padding=ft.Padding.symmetric(horizontal=10, vertical=8),
                    bgcolor=ft.Colors.TRANSPARENT,
                ),
            ],
            spacing=4,
        )
    if isinstance(value, np.ndarray):
        if value.ndim == 2:
            return ft.Column(
                [
                    ft.Text(title, weight=ft.FontWeight.W_600),
                    ft.Container(
                        content=latex_matrix(value, fontsize=LATEX_SIZE_MEDIUM),
                        padding=ft.Padding.symmetric(horizontal=10, vertical=8),
                        bgcolor=ft.Colors.TRANSPARENT,
                        border_radius=8,
                    ),
                ],
                spacing=4,
            )
        return ft.Column(
            [
                ft.Text(title, weight=ft.FontWeight.W_600),
                ft.Container(
                    content=latex_vector(value, fontsize=LATEX_SIZE_MEDIUM, column=True),
                    padding=ft.Padding.symmetric(horizontal=10, vertical=6),
                    bgcolor=ft.Colors.TRANSPARENT,
                    border_radius=8,
                ),
            ],
            spacing=4,
        )
    if isinstance(value, (list, tuple)) and value and all(isinstance(v, np.ndarray) for v in value):
        items: list[ft.Control] = [ft.Text(title, weight=ft.FontWeight.W_600)]
        for i, arr in enumerate(value, 1):
            arr = np.asarray(arr)
            items.append(
                ft.Container(
                    content=ft.Row(
                        [
                            ft.Text(f"[{i}]", weight=ft.FontWeight.W_600),
                            latex_vector(arr, fontsize=LATEX_SIZE_MEDIUM, column=False),
                        ],
                        spacing=8,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    padding=ft.Padding.symmetric(horizontal=10, vertical=6),
                    bgcolor=ft.Colors.TRANSPARENT,
                    border_radius=8,
                )
            )
        return ft.Column(items, spacing=4)
    if isinstance(value, (float, np.floating, int, np.integer)):
        return _latex_scalar_block(label, float(value))
    return ft.Row(
        [ft.Text(f"{title}:", weight=ft.FontWeight.W_600), ft.Text(fmt(value), selectable=True)],
        spacing=8,
    )


def render_result(
    title: str,
    result: dict[str, Any],
    values: tuple[str, ...] = (),
    tables: tuple[str, ...] = (),
    extras: list[ft.Control] | None = None,
) -> ft.Control:
    blocks: list[ft.Control] = []
    status = result.get("status", "unknown")
    status_color = ft.Colors.GREEN_700 if status == "ok" else ft.Colors.RED_700
    blocks.append(
        ft.Row(
            [
                ft.Text("Статус:", weight=ft.FontWeight.W_600),
                ft.Text("успешно" if status == "ok" else "ошибка", color=status_color, weight=ft.FontWeight.W_600),
            ],
            spacing=8,
        )
    )
    if result.get("message"):
        blocks.append(ft.Text(f"Сообщение: {result['message']}", italic=True))

    if status == "ok":
        for key in values:
            if key in result:
                blocks.append(value_block(key, result[key]))
        for key in tables:
            df = result.get(key)
            if isinstance(df, pd.DataFrame):
                blocks.append(
                    ft.Column(
                        [ft.Text(LABELS.get(key, key), weight=ft.FontWeight.W_600), df_to_datatable(df)],
                        spacing=6,
                    )
                )
        if extras:
            blocks.extend(extras)
    return card(title, *blocks)


# ===== Ввод СЛАУ как системы уравнений =====

class SystemMatrixInput:
    def __init__(self, n: int, field_width: int = 78):
        self.n = n
        self.field_width = field_width
        self.A_fields: list[list[ft.TextField]] = []
        self.b_fields: list[ft.TextField] = []
        self._build_fields(n)

    def _build_fields(self, n: int) -> None:
        self.n = n
        self.A_fields = []
        self.b_fields = []
        for _ in range(n):
            row = [self._mk_field(self.field_width) for _ in range(n)]
            self.A_fields.append(row)
            self.b_fields.append(self._mk_field(self.field_width))

    @staticmethod
    def _mk_field(width: int) -> ft.TextField:
        return ft.TextField(
            value="",
            text_align=ft.TextAlign.RIGHT,
            width=width,
            height=40,
            text_size=14,
            content_padding=ft.Padding.symmetric(horizontal=6, vertical=4),
            dense=True,
        )

    def snapshot(self) -> tuple[list[list[str]], list[str]]:
        A = [[cell.value or "" for cell in row] for row in self.A_fields]
        b = [cell.value or "" for cell in self.b_fields]
        return A, b

    def resize(self, n: int) -> None:
        old_A, old_b = self.snapshot()
        self._build_fields(n)
        for i in range(min(n, len(old_A))):
            for j in range(min(n, len(old_A[i]))):
                self.A_fields[i][j].value = old_A[i][j]
            if i < len(old_b):
                self.b_fields[i].value = old_b[i]

    def build(self) -> ft.Control:
        eq_rows: list[ft.Control] = []
        for i in range(self.n):
            parts: list[ft.Control] = []
            for j in range(self.n):
                parts.append(self.A_fields[i][j])
                parts.append(latex_inline(rf"\cdot x_{{{j + 1}}}", fontsize=LATEX_SIZE_SMALL))
                if j < self.n - 1:
                    parts.append(latex_inline("+", fontsize=LATEX_SIZE_SMALL))
            parts.append(latex_inline("=", fontsize=LATEX_SIZE_SMALL))
            parts.append(self.b_fields[i])
            eq_rows.append(
                ft.Row(
                    parts,
                    spacing=6,
                    alignment=ft.MainAxisAlignment.START,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    scroll=ft.ScrollMode.AUTO,
                )
            )
        brace_size = max(72, 38 * self.n)
        return ft.Container(
            content=ft.Row(
                [
                    ft.Container(
                        content=ft.Text(
                            "{",
                            size=brace_size,
                            font_family="Cambria Math",
                            color=AXIS_COLOR,
                            height=0.86,
                        ),
                        width=28,
                        alignment=ft.Alignment.CENTER,
                    ),
                    ft.Column(eq_rows, spacing=10, scroll=ft.ScrollMode.AUTO),
                ],
                spacing=8,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                scroll=ft.ScrollMode.AUTO,
            ),
            bgcolor=ft.Colors.TRANSPARENT,
            border_radius=10,
            padding=ft.Padding.symmetric(horizontal=14, vertical=12),
        )

    def set_values(self, A: list[list[float]], b: list[float]) -> None:
        if len(A) != self.n:
            self._build_fields(len(A))
        for i in range(self.n):
            for j in range(self.n):
                self.A_fields[i][j].value = f"{A[i][j]:.4f}"
            self.b_fields[i].value = f"{b[i]:.4f}"

    def get_A(self) -> np.ndarray:
        return np.array(
            [[float((c.value or "0").replace(",", ".")) for c in row] for row in self.A_fields],
            dtype=float,
        )

    def get_b(self) -> np.ndarray:
        return np.array(
            [float((c.value or "0").replace(",", ".")) for c in self.b_fields],
            dtype=float,
        )


def solution_system_view(values: np.ndarray, prefix: str = "x", title: str = "Ответ") -> ft.Control:
    flat = np.asarray(values).flatten()
    lines = [rf"{prefix}_{{{i + 1}}} = {latex_num(v)}" for i, v in enumerate(flat)]
    body = latex_braced_system(lines, fontsize=LATEX_SIZE_LARGE)
    return ft.Column(
        [
            ft.Text(title, weight=ft.FontWeight.W_600),
            ft.Container(
                content=body,
                padding=ft.Padding.symmetric(horizontal=12, vertical=10),
                bgcolor=ft.Colors.SURFACE_CONTAINER,
                border_radius=10,
                border=ft.Border(
                    left=ft.BorderSide(width=3, color=TERTIARY),
                    right=ft.BorderSide(width=3, color=TERTIARY),
                    top=ft.BorderSide(width=0.5, color=ft.Colors.OUTLINE_VARIANT),
                    bottom=ft.BorderSide(width=0.5, color=ft.Colors.OUTLINE_VARIANT),
                ),
            ),
        ],
        spacing=6,
    )


def local_quadratic_derivatives(
    x_values: np.ndarray,
    y_values: np.ndarray,
    points: dict[str, float],
) -> pd.DataFrame:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    rows: list[dict[str, Any]] = []
    if x.size < 3 or y.size < 3:
        return pd.DataFrame(rows)

    for label, point in points.items():
        nearest_right = int(np.searchsorted(x, point))
        start = max(0, min(nearest_right - 1, len(x) - 3))
        selected = slice(start, start + 3)
        coefficients = np.polyfit(x[selected], y[selected], deg=2)
        rows.append(
            {
                "point": label,
                "x": point,
                "nodes": ", ".join(latex_num(value, precision=4) for value in x[selected]),
                "y'": np.polyval(np.polyder(coefficients, 1), point),
                "y''": np.polyval(np.polyder(coefficients, 2), point),
            }
        )
    return pd.DataFrame(rows)


def solution_xy_table(result: dict[str, Any]) -> pd.DataFrame:
    xs = np.asarray(result.get("x", []), dtype=float)
    ys = np.asarray(result.get("y", []), dtype=float)
    count = min(xs.size, ys.size)
    return pd.DataFrame({"x": xs[:count], "y": ys[:count]})


# ===== Ввод векторов и таблиц =====

class VectorInput:
    def __init__(self, size: int, width: int = 90):
        self.size = size
        self.fields: list[ft.TextField] = [
            ft.TextField(
                value="",
                text_align=ft.TextAlign.RIGHT,
                width=width,
                height=42,
                text_size=13,
                content_padding=ft.Padding.symmetric(horizontal=6, vertical=4),
            )
            for _ in range(size)
        ]

    def build(self, horizontal: bool = True) -> ft.Control:
        if horizontal:
            return ft.Row(self.fields, spacing=4)
        return ft.Column(self.fields, spacing=4)

    def set_values(self, values: list[float]) -> None:
        for i, v in enumerate(values):
            self.fields[i].value = f"{v:.4f}"

    def get(self) -> np.ndarray:
        return np.array(
            [float((c.value or "0").replace(",", ".")) for c in self.fields],
            dtype=float,
        )


class XYTableInput:
    def __init__(self, size: int):
        self.x = VectorInput(size, width=86)
        self.y = VectorInput(size, width=86)

    def build(self) -> ft.Control:
        return ft.Column(
            [
                ft.Row([ft.Container(ft.Text("xᵢ", weight=ft.FontWeight.W_600), width=24), self.x.build()], spacing=4),
                ft.Row([ft.Container(ft.Text("yᵢ", weight=ft.FontWeight.W_600), width=24), self.y.build()], spacing=4),
            ],
            spacing=6,
            scroll=ft.ScrollMode.AUTO,
        )

    def set_values(self, xs: list[float], ys: list[float]) -> None:
        self.x.set_values(xs)
        self.y.set_values(ys)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([["x", *self.x.get().tolist()], ["y", *self.y.get().tolist()]])

    def get_xy(self) -> tuple[np.ndarray, np.ndarray]:
        return self.x.get(), self.y.get()


# ===== Утилиты =====

def show_error(page: ft.Page, message: str) -> None:
    page.open(ft.SnackBar(content=ft.Text(f"Ошибка: {message}"), bgcolor=ft.Colors.RED_700))


def results_container() -> ft.Column:
    return ft.Column(spacing=14, scroll=ft.ScrollMode.AUTO)


def action_row(*buttons: ft.Control) -> ft.Row:
    return ft.Row(list(buttons), spacing=10, wrap=True)


def variant_button(on_click) -> ft.Control:
    return ft.OutlinedButton("Вариант 6", icon=ft.Icons.DOWNLOAD, on_click=on_click)


def solve_button(on_click, label: str = "Решить") -> ft.Control:
    return ft.FilledButton(label, icon=ft.Icons.PLAY_ARROW, on_click=on_click)


def secondary_button(label: str, icon: str, on_click) -> ft.Control:
    return ft.OutlinedButton(label, icon=icon, on_click=on_click)


def dimension_dropdown(value: int, on_select, min_n: int = 2, max_n: int = 6) -> ft.Dropdown:
    return ft.Dropdown(
        label="Размерность",
        value=str(value),
        width=150,
        dense=True,
        options=[
            ft.DropdownOption(key=str(n), text=f"{n} x {n}")
            for n in range(min_n, max_n + 1)
        ],
        on_select=on_select,
    )


def rail_number_icon(number: int, selected: bool = False) -> ft.Control:
    return ft.Container(
        width=24,
        height=24,
        alignment=ft.Alignment.CENTER,
        bgcolor=PRIMARY if selected else "#444753",
        border_radius=5,
        content=ft.Text(
            str(number),
            size=11 if number < 10 else 9,
            weight=ft.FontWeight.W_700,
            color=ft.Colors.WHITE,
            text_align=ft.TextAlign.CENTER,
        ),
    )


# ===== LineChart helpers =====

@dataclass
class ChartSeries:
    xs: list[float]
    ys: list[float]
    color: str = PRIMARY
    stroke_width: float = 2.5
    curved: bool = False
    dash_pattern: list[int] | None = None
    show_points: bool = False
    point_size: float = 5.0
    point_color: str | None = None
    below_fill: str | None = None
    below_cutoff_y: float | None = None


def _sample(f: Callable[[float], float], a: float, b: float, n: int = 200) -> tuple[list[float], list[float]]:
    xs_arr = np.linspace(a, b, n)
    ys: list[float] = []
    xs: list[float] = []
    for x in xs_arr:
        try:
            y = f(float(x))
            if not np.isfinite(y):
                continue
            xs.append(float(x))
            ys.append(float(y))
        except Exception:
            continue
    return xs, ys


def _series_to_line(series: ChartSeries) -> fch.LineChartData:
    points = [fch.LineChartDataPoint(x, y) for x, y in zip(series.xs, series.ys)]
    point_marker = None
    if series.show_points:
        point_marker = fch.ChartCirclePoint(
            radius=series.point_size,
            color=series.point_color or series.color,
            stroke_color=ft.Colors.WHITE,
            stroke_width=1.5,
        )
    return fch.LineChartData(
        points=points,
        color=series.color,
        stroke_width=series.stroke_width,
        curved=series.curved,
        rounded_stroke_cap=True,
        dash_pattern=series.dash_pattern,
        point=point_marker if point_marker is not None else False,
        below_line_bgcolor=series.below_fill,
        below_line_cutoff_y=series.below_cutoff_y,
    )


def _finite_values(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _pad_axis(data_min: float, data_max: float, pad_frac: float) -> tuple[float, float]:
    if data_min == data_max:
        if data_min == 0:
            data_min, data_max = 0.0, 1.0
        else:
            delta = max(abs(data_min) * 0.12, 1e-3)
            data_min -= delta
            data_max += delta
    span = data_max - data_min
    pad = span * pad_frac
    lower = data_min - pad
    upper = data_max + pad
    if upper <= lower:
        upper = lower + max(abs(lower) * 0.12, 1.0)
    return lower, upper


def _compute_bounds(
    series_list: list[ChartSeries],
    pad_frac: float = 0.08,
    x_pad_frac: float | None = None,
    y_pad_frac: float | None = None,
) -> tuple[float, float, float, float]:
    if x_pad_frac is None:
        x_pad_frac = pad_frac
    if y_pad_frac is None:
        y_pad_frac = pad_frac
    xs_all: list[float] = []
    ys_all: list[float] = []
    for s in series_list:
        xs_all.extend(s.xs)
        ys_all.extend(s.ys)
    xs_arr = _finite_values(xs_all)
    ys_arr = _finite_values(ys_all)
    if xs_arr.size == 0 or ys_arr.size == 0:
        return 0.0, 1.0, 0.0, 1.0
    x_min, x_max = _pad_axis(float(xs_arr.min()), float(xs_arr.max()), x_pad_frac)
    y_min, y_max = _pad_axis(float(ys_arr.min()), float(ys_arr.max()), y_pad_frac)
    return x_min, x_max, y_min, y_max


AXIS_COLOR = "#0F172A"
GRID_COLOR = ft.Colors.with_opacity(0.18, "#475569")


def _nice_step(span: float, target_ticks: int = 5) -> float:
    """Подбор «красивого» шага для оси из множества 1, 2, 5 × 10^k."""
    if span <= 0:
        return 1.0
    raw = span / target_ticks
    magnitude = 10 ** np.floor(np.log10(raw))
    fraction = raw / magnitude
    if fraction < 1.5:
        nice = 1
    elif fraction < 3:
        nice = 2
    elif fraction < 7:
        nice = 5
    else:
        nice = 10
    return float(nice * magnitude)


def _nice_ticks(min_v: float, max_v: float, target: int = 5) -> list[float]:
    if max_v <= min_v:
        return [min_v]
    step = _nice_step(max_v - min_v, target)
    start = np.ceil(min_v / step) * step
    end = np.floor(max_v / step) * step
    ticks: list[float] = []
    t = start
    while t <= end + step * 1e-6:
        ticks.append(float(t))
        t += step
    return ticks


def _format_tick(v: float) -> str:
    if abs(v) < 1e-12:
        return "0"
    if abs(v) >= 1e4 or abs(v) < 1e-3:
        return f"{v:.2e}"
    return f"{v:g}"


def _axis_labels(min_v: float, max_v: float, count: int = 5) -> list[fch.ChartAxisLabel]:
    """Красивые равномерные подписи оси (шаги 1/2/5 × 10^k)."""
    if max_v <= min_v:
        return []
    ticks = _nice_ticks(min_v, max_v, count)
    out: list[fch.ChartAxisLabel] = []
    for t in ticks:
        out.append(
            fch.ChartAxisLabel(
                value=float(t),
                label=ft.Text(
                    _format_tick(t),
                    size=11,
                    color=AXIS_COLOR,
                    weight=ft.FontWeight.W_600,
                ),
            )
        )
    return out


def _axis_interval(min_v: float, max_v: float) -> float:
    if max_v <= min_v:
        return 1.0
    return _nice_step(max_v - min_v)


def _is_zero_axis_series(series: ChartSeries) -> bool:
    return (
        len(series.xs) == 2
        and len(series.ys) == 2
        and all(abs(y) < 1e-12 for y in series.ys)
        and series.color == AXIS_COLOR
        and series.dash_pattern is None
        and not series.show_points
    )


def _stretch_zero_axis(series: list[ChartSeries], x_min: float, x_max: float) -> list[ChartSeries]:
    return [
        replace(s, xs=[x_min, x_max])
        if _is_zero_axis_series(s)
        else s
        for s in series
    ]


def _with_visible_zero_axis(
    series: list[ChartSeries],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> list[ChartSeries]:
    stretched = _stretch_zero_axis(series, x_min, x_max)
    if y_min <= 0.0 <= y_max and not any(_is_zero_axis_series(s) for s in stretched):
        return [*stretched, _zero_line(x_min, x_max)]
    return stretched


def _apply_line_chart_view(
    chart: fch.LineChart,
    series: list[ChartSeries],
    x_pad_frac: float = 0.08,
    y_pad_frac: float = 0.08,
) -> None:
    x_min, x_max, y_min, y_max = _compute_bounds(
        series,
        x_pad_frac=x_pad_frac,
        y_pad_frac=y_pad_frac,
    )
    series = _with_visible_zero_axis(series, x_min, x_max, y_min, y_max)
    chart.data_series = [_series_to_line(s) for s in series]
    chart.min_x, chart.max_x = x_min, x_max
    chart.min_y, chart.max_y = y_min, y_max
    chart.left_axis = fch.ChartAxis(label_size=48, labels=_axis_labels(y_min, y_max))
    chart.bottom_axis = fch.ChartAxis(label_size=28, labels=_axis_labels(x_min, x_max))
    chart.horizontal_grid_lines = fch.ChartGridLines(
        interval=_axis_interval(y_min, y_max),
        color=GRID_COLOR,
        width=0.8,
        dash_pattern=[3, 4],
    )
    chart.vertical_grid_lines = fch.ChartGridLines(
        interval=_axis_interval(x_min, x_max),
        color=GRID_COLOR,
        width=0.8,
        dash_pattern=[3, 4],
    )


def build_line_chart(
    series: list[ChartSeries],
    height: int = 380,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    x_pad_frac: float = 0.08,
    y_pad_frac: float = 0.08,
) -> fch.LineChart:
    auto_xmin, auto_xmax, auto_ymin, auto_ymax = _compute_bounds(
        series,
        x_pad_frac=x_pad_frac,
        y_pad_frac=y_pad_frac,
    )
    if x_min is None:
        x_min = auto_xmin
    if x_max is None:
        x_max = auto_xmax
    if y_min is None:
        y_min = auto_ymin
    if y_max is None:
        y_max = auto_ymax

    series = _with_visible_zero_axis(series, x_min, x_max, y_min, y_max)

    return fch.LineChart(
        data_series=[_series_to_line(s) for s in series],
        animation=0,
        min_x=x_min,
        max_x=x_max,
        min_y=y_min,
        max_y=y_max,
        expand=True,
        border=ft.Border(
            bottom=ft.BorderSide(0.6, ft.Colors.with_opacity(0.28, AXIS_COLOR)),
            left=ft.BorderSide(1.8, AXIS_COLOR),
            top=ft.BorderSide(0.6, ft.Colors.with_opacity(0.28, AXIS_COLOR)),
            right=ft.BorderSide(0.6, ft.Colors.with_opacity(0.28, AXIS_COLOR)),
        ),
        horizontal_grid_lines=fch.ChartGridLines(
            interval=_axis_interval(y_min, y_max),
            color=GRID_COLOR,
            width=0.8,
            dash_pattern=[3, 4],
        ),
        vertical_grid_lines=fch.ChartGridLines(
            interval=_axis_interval(x_min, x_max),
            color=GRID_COLOR,
            width=0.8,
            dash_pattern=[3, 4],
        ),
        left_axis=fch.ChartAxis(label_size=48, labels=_axis_labels(y_min, y_max)),
        bottom_axis=fch.ChartAxis(label_size=28, labels=_axis_labels(x_min, x_max)),
        top_axis=fch.ChartAxis(show_labels=False),
        right_axis=fch.ChartAxis(show_labels=False),
        interactive=True,
    )


def chart_legend(items: list[tuple[str, str]]) -> ft.Control:
    """Каждый элемент: (LaTeX-строка, цвет)."""
    chips: list[ft.Control] = []
    for label, color in items:
        chips.append(
            ft.Row(
                [
                    ft.Container(width=14, height=14, bgcolor=color, border_radius=4),
                    latex_inline(label, fontsize=LATEX_SIZE_TINY),
                ],
                spacing=6,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            )
        )
    return ft.Row(chips, wrap=True, spacing=18)


def chart_card(title: str, chart: ft.Control, legend: ft.Control | None = None, height: int = 380) -> ft.Control:
    children: list[ft.Control] = [ft.Container(content=chart, height=height)]
    if legend is not None:
        children.append(legend)
    return card(title, *children)


# ===== Базовые серии графиков =====

def _zero_line(x_min: float, x_max: float) -> ChartSeries:
    return ChartSeries(
        xs=[x_min, x_max], ys=[0.0, 0.0],
        color=AXIS_COLOR, stroke_width=1.8,
    )


def series_function(f: Callable[[float], float], a: float, b: float, color: str = PRIMARY) -> ChartSeries:
    xs, ys = _sample(f, a, b)
    return ChartSeries(xs=xs, ys=ys, color=color, stroke_width=2.8, curved=False)


def series_marker(x: float, y: float, color: str = ACCENT, size: float = 8.0) -> ChartSeries:
    return ChartSeries(xs=[x], ys=[y], color=color, stroke_width=0, show_points=True, point_size=size)


def series_segment(x0: float, y0: float, x1: float, y1: float, color: str, dashed: bool = False) -> ChartSeries:
    return ChartSeries(
        xs=[x0, x1], ys=[y0, y1],
        color=color, stroke_width=2,
        dash_pattern=[6, 4] if dashed else None,
    )


def _focus_range(
    values: list[float],
    pad_frac: float = 0.3,
    min_span: float = 1e-3,
    domain: tuple[float, float] | None = None,
) -> tuple[float, float]:
    finite = _finite_values(values)
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = float(finite.min()), float(finite.max())
    span = max(hi - lo, min_span)
    center = (lo + hi) / 2
    half = span * (0.5 + pad_frac)
    left, right = center - half, center + half
    if domain is not None:
        d0, d1 = sorted(domain)
        left = max(left, d0)
        right = min(right, d1)
        if right <= left:
            left, right = d0, d1
    return left, right


def _zero_line_for_view(x_min: float, x_max: float) -> ChartSeries:
    return _zero_line(x_min, x_max)


# ===== Графики для задания 4 =====

def chart_function_only(f: Callable, a: float, b: float) -> tuple[ft.Control, ft.Control]:
    s_f = series_function(f, a, b, color=PRIMARY)
    s_zero = _zero_line_for_view(a, b)
    chart = build_line_chart([s_f, s_zero], height=380, x_pad_frac=0.0)
    legend = chart_legend([("f(x)", PRIMARY), ("y = 0", AXIS_COLOR)])
    return chart, legend


def chart_mpi_step(f: Callable, a: float, b: float, iter_df: pd.DataFrame, k: int) -> tuple[list[ChartSeries], list[tuple[str, str]]]:
    row = iter_df.iloc[k]
    x_n = float(row["x_n"])
    x_next = float(row["x_n+1"])
    min_span = max(abs(b - a) * 0.04, 1e-3)
    focus_x0, focus_x1 = _focus_range([x_n, x_next], pad_frac=0.6, min_span=min_span, domain=(a, b))
    series = [
        series_function(f, focus_x0, focus_x1, color=PRIMARY),
        _zero_line_for_view(focus_x0, focus_x1),
        series_marker(x_n, f(x_n), color=ACCENT, size=9),
        series_marker(x_next, f(x_next), color=TERTIARY, size=8),
    ]
    legend = [
        ("f(x)", PRIMARY),
        (rf"x_n\;(\text{{шаг}}\;{k + 1})", ACCENT),
        ("x_{n+1}", TERTIARY),
    ]
    return series, legend


def chart_dihotomy_step(f: Callable, iter_df: pd.DataFrame, k: int) -> tuple[list[ChartSeries], list[tuple[str, str]]]:
    a0 = float(iter_df.iloc[0]["a"])
    b0 = float(iter_df.iloc[0]["b"])
    row = iter_df.iloc[k]
    a_k = float(row["a"])
    b_k = float(row["b"])
    x_0 = float(row["x_0"])
    min_span = max(abs(b0 - a0) * 0.04, 1e-3)
    focus_x0, focus_x1 = _focus_range([a_k, b_k, x_0], pad_frac=0.18, min_span=min_span, domain=(a0, b0))
    fa, fb = f(a_k), f(b_k)
    fx0 = f(x_0)
    span = max(abs(fa), abs(fb), abs(fx0), 0.1)
    series = [
        series_function(f, focus_x0, focus_x1, color=PRIMARY),
        _zero_line_for_view(focus_x0, focus_x1),
        series_segment(a_k, -span, a_k, span, color=SECONDARY, dashed=False),
        series_segment(b_k, -span, b_k, span, color=SECONDARY, dashed=False),
        series_marker(x_0, fx0, color=ACCENT, size=10),
    ]
    legend = [
        ("f(x)", PRIMARY),
        (rf"[a,\,b]\;(\text{{шаг}}\;{k + 1})", SECONDARY),
        (r"x_0 = \frac{a+b}{2}", ACCENT),
    ]
    return series, legend


def chart_newton_step(f: Callable, a: float, b: float, iter_df: pd.DataFrame, k: int) -> tuple[list[ChartSeries], list[tuple[str, str]]]:
    row = iter_df.iloc[k]
    x_n = float(row["x_n"])
    f_xn = float(row["f(x_n)"])
    df_xn = float(row["f'(x_n)"])
    x_next = float(row["x_n+1"])
    min_span = max(abs(b - a) * 0.04, 1e-3)
    t_x0, t_x1 = _focus_range([x_n, x_next], pad_frac=0.75, min_span=min_span, domain=(a, b))
    t_y0 = f_xn + df_xn * (t_x0 - x_n)
    t_y1 = f_xn + df_xn * (t_x1 - x_n)
    series = [
        series_function(f, t_x0, t_x1, color=PRIMARY),
        _zero_line_for_view(t_x0, t_x1),
        series_segment(t_x0, t_y0, t_x1, t_y1, color=SECONDARY, dashed=True),
        series_segment(x_n, 0.0, x_n, f_xn, color=ACCENT, dashed=True),
        series_marker(x_n, f_xn, color=ACCENT, size=9),
        series_marker(x_next, f(x_next), color=TERTIARY, size=8),
    ]
    legend = [
        ("f(x)", PRIMARY),
        (r"\text{касательная}", SECONDARY),
        (rf"x_n\;(\text{{шаг}}\;{k + 1})", ACCENT),
        ("x_{n+1}", TERTIARY),
    ]
    return series, legend


# ===== Графики для остальных задач =====

def _lagrange_eval(x_nodes: np.ndarray, y_nodes: np.ndarray, x: float) -> float:
    n = len(x_nodes)
    total = 0.0
    for i in range(n):
        term = float(y_nodes[i])
        for j in range(n):
            if j != i:
                term *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        total += term
    return total


def chart_interpolation(
    xs: np.ndarray, ys: np.ndarray,
    x_star: float | None,
    lagrange_val: float | None,
    newton_val: float | None,
) -> tuple[ft.Control, ft.Control]:
    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    grid = np.linspace(x_min, x_max, 300)
    curve_ys = np.array([_lagrange_eval(xs, ys, gx) for gx in grid], dtype=float)
    series: list[ChartSeries] = [
        ChartSeries(
            xs=grid.tolist(), ys=curve_ys.tolist(),
            color=PRIMARY, stroke_width=2.4, curved=False,
        ),
        ChartSeries(
            xs=xs.tolist(), ys=ys.tolist(),
            color=PRIMARY_DARK, stroke_width=0,
            show_points=True, point_size=7,
        ),
    ]
    legend: list[tuple[str, str]] = [
        ("L(x) = N(x)", PRIMARY),
        (r"\text{узлы}", PRIMARY_DARK),
    ]
    if x_star is not None and lagrange_val is not None:
        series.append(series_marker(x_star, lagrange_val, color=ACCENT, size=11))
        legend.append((rf"L(x^{{*}}) = {latex_num(lagrange_val)}", ACCENT))
    if x_star is not None and newton_val is not None and abs((newton_val or 0) - (lagrange_val or 0)) > 1e-9:
        series.append(series_marker(x_star, newton_val, color=SECONDARY, size=9))
        legend.append((rf"N(x^{{*}}) = {latex_num(newton_val)}", SECONDARY))
    chart = build_line_chart(series, height=380)
    return chart, chart_legend(legend)


def chart_spline(xs: np.ndarray, ys: np.ndarray, spline_result: dict[str, Any]) -> tuple[ft.Control, ft.Control]:
    coeffs = np.asarray(spline_result.get("coefficients", []))
    curve_x: list[float] = []
    curve_y: list[float] = []
    segment_count = min(len(xs) - 1, len(coeffs))
    for i in range(segment_count):
        a, b, c, d = coeffs[i]
        x_left, x_right = float(xs[i]), float(xs[i + 1])
        seg = np.linspace(x_left, x_right, 80)
        dx = seg - x_left
        vals = a + b * dx + c * dx ** 2 + d * dx ** 3
        if i > 0:
            seg = seg[1:]
            vals = vals[1:]
        curve_x.extend(seg.tolist())
        curve_y.extend(vals.tolist())
    series = [
        ChartSeries(
            xs=curve_x, ys=curve_y,
            color=PRIMARY, stroke_width=2.6, curved=False,
        ),
        ChartSeries(
            xs=xs.tolist(), ys=ys.tolist(),
            color=ACCENT, stroke_width=0,
            show_points=True, point_size=7,
        ),
    ]
    chart = build_line_chart(series, height=400)
    legend = chart_legend([(r"\text{сплайн}", PRIMARY), (r"\text{узлы}", ACCENT)])
    return chart, legend


def chart_least_squares(
    xs: np.ndarray, ys: np.ndarray,
    a1: float | None, b1: float | None,
    a2: float | None, b2: float | None, c2: float | None,
) -> tuple[ft.Control, ft.Control]:
    x_min = float(np.min(xs))
    x_max = float(np.max(xs))
    grid = np.linspace(x_min, x_max, 200)
    series: list[ChartSeries] = [
        ChartSeries(
            xs=xs.tolist(), ys=ys.tolist(),
            color=PRIMARY, stroke_width=0,
            show_points=True, point_size=7,
        ),
    ]
    legend: list[tuple[str, str]] = [(r"\text{данные}", PRIMARY)]
    if a1 is not None and b1 is not None:
        series.append(ChartSeries(
            xs=grid.tolist(), ys=(a1 * grid + b1).tolist(),
            color=ACCENT, stroke_width=2.6,
        ))
        legend.append(("P_1(x) = a\\,x + b", ACCENT))
    if a2 is not None and b2 is not None and c2 is not None:
        series.append(ChartSeries(
            xs=grid.tolist(), ys=(a2 * grid ** 2 + b2 * grid + c2).tolist(),
            color=SECONDARY, stroke_width=2.6,
        ))
        legend.append(("P_2(x) = a\\,x^2 + b\\,x + c", SECONDARY))
    chart = build_line_chart(series, height=420)
    return chart, chart_legend(legend)


def chart_simpson(f: Callable, a: float, b: float) -> tuple[ft.Control, ft.Control]:
    xs, ys = _sample(f, a, b, n=200)
    fill_color = ft.Colors.with_opacity(0.22, PRIMARY)
    series = [
        ChartSeries(
            xs=xs, ys=ys,
            color=PRIMARY, stroke_width=2.6,
            below_fill=fill_color,
            below_cutoff_y=0.0,
        ),
        _zero_line(a, b),
    ]
    chart = build_line_chart(series, height=400)
    legend = chart_legend([("f(x)", PRIMARY), (r"\int_a^b f(x)\,dx", fill_color)])
    return chart, legend


def chart_cauchy(rk_result: dict, adams_result: dict) -> tuple[ft.Control, ft.Control]:
    series: list[ChartSeries] = []
    legend: list[tuple[str, str]] = []
    if rk_result.get("status") == "ok":
        xs = np.asarray(rk_result["x"], dtype=float)
        ys = np.asarray(rk_result["y"], dtype=float)
        series.append(ChartSeries(
            xs=xs.tolist(), ys=ys.tolist(),
            color=PRIMARY, stroke_width=2.8,
            show_points=True, point_size=5,
        ))
        legend.append((r"\text{Рунге-Кутта 4}", PRIMARY))
    if adams_result.get("status") == "ok":
        xs = np.asarray(adams_result["x"], dtype=float)
        ys = np.asarray(adams_result["y"], dtype=float)
        series.append(ChartSeries(
            xs=xs.tolist(), ys=ys.tolist(),
            color=ACCENT, stroke_width=2.4,
            dash_pattern=[6, 4],
            show_points=True, point_size=5,
        ))
        legend.append((r"\text{Адамс}", ACCENT))
    chart = build_line_chart(series, height=420)
    return chart, chart_legend(legend)


def chart_boundary(progonka_result: dict) -> tuple[ft.Control, ft.Control]:
    xs = np.asarray(progonka_result["x"], dtype=float)
    ys = np.asarray(progonka_result["y"], dtype=float)
    series = [
        ChartSeries(
            xs=xs.tolist(), ys=ys.tolist(),
            color=PRIMARY, stroke_width=2.8,
            show_points=True, point_size=6,
        )
    ]
    chart = build_line_chart(series, height=400)
    return chart, chart_legend([("y(x)", PRIMARY)])


# ===== Анимация итераций (Flet slider) =====

def iteration_player(
    title: str,
    n_steps: int,
    step_builder: Callable[[int], tuple[list[ChartSeries], list[tuple[str, str]]]],
    page: ft.Page,
    x_pad_frac: float = 0.08,
    y_pad_frac: float = 0.08,
) -> ft.Control:
    if n_steps <= 0:
        return ft.Container()

    initial_series, initial_legend = step_builder(0)
    chart = build_line_chart(
        initial_series,
        height=400,
        x_pad_frac=x_pad_frac,
        y_pad_frac=y_pad_frac,
    )
    legend_holder = ft.Container(content=chart_legend(initial_legend))
    step_label = ft.Text(f"Шаг 1 / {n_steps}", weight=ft.FontWeight.W_600)

    def render_step(k: int) -> None:
        new_series, new_legend = step_builder(k)
        _apply_line_chart_view(
            chart,
            new_series,
            x_pad_frac=x_pad_frac,
            y_pad_frac=y_pad_frac,
        )
        chart.update()
        legend_holder.content = chart_legend(new_legend)
        legend_holder.update()
        step_label.value = f"Шаг {k + 1} / {n_steps}"
        step_label.update()

    def on_slider_change(e: ft.ControlEvent) -> None:
        render_step(int(e.control.value))

    def step_prev(_) -> None:
        k = max(0, int(slider.value) - 1)
        slider.value = k
        slider.update()
        render_step(k)

    def step_next(_) -> None:
        k = min(n_steps - 1, int(slider.value) + 1)
        slider.value = k
        slider.update()
        render_step(k)

    slider = ft.Slider(
        min=0, max=n_steps - 1, divisions=max(1, n_steps - 1),
        value=0, label="{value}",
        on_change=on_slider_change, expand=True,
    )

    return card(
        title,
        ft.Container(content=chart, height=400),
        legend_holder,
        ft.Row(
            [
                ft.IconButton(icon=ft.Icons.SKIP_PREVIOUS, on_click=step_prev, tooltip="Назад"),
                slider,
                ft.IconButton(icon=ft.Icons.SKIP_NEXT, on_click=step_next, tooltip="Вперёд"),
            ],
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        step_label,
    )


# ===== Страницы =====

def page_home() -> ft.Control:
    return ft.Column(
        [ft.Text("Численные методы", size=28, weight=ft.FontWeight.BOLD)],
        spacing=14,
        scroll=ft.ScrollMode.AUTO,
    )


def page_task1(page: ft.Page) -> ft.Control:
    system = SystemMatrixInput(3)
    system_holder = ft.Container(content=system.build())
    eps = ft.TextField(label="Точность ε", value="", width=160)
    results = results_container()

    def change_dimension(e: ft.ControlEvent) -> None:
        system.resize(int(e.control.value or system.n))
        system_holder.content = system.build()
        results.controls.clear()
        page.update()

    dimension = dimension_dropdown(3, change_dimension)

    def load_variant(_):
        dimension.value = "3"
        system.resize(3)
        system.set_values(
            [[1.61, 0.71, -0.05], [-1.03, -2.05, 0.87], [2.50, -3.12, -6.03]],
            [0.44, -1.16, -7.50],
        )
        system_holder.content = system.build()
        eps.value = "0.01"
        page.update()

    def solve(_):
        results.controls.clear()
        try:
            A = system.get_A()
            b = system.get_b()
            ev = float((eps.value or "0.01").replace(",", "."))
            gauss = pretty_solve_gauss_matrix(A, b)
            mpi = pretty_solve_mpi_slay(ev, A, b)

            gauss_extras: list[ft.Control] = []
            if gauss.get("status") == "ok":
                gauss_extras.append(solution_system_view(gauss["solution"], "x", "Ответ"))
            results.controls.append(
                render_result("A) Метод Гаусса", gauss,
                              values=("residual", "residual_norm_inf", "matrix_after_forward_pass"),
                              extras=gauss_extras)
            )
            mpi_extras: list[ft.Control] = []
            if mpi.get("status") == "ok":
                mpi_extras.append(solution_system_view(mpi["solutions"], "x", "Ответ"))
            results.controls.append(
                render_result("B) Метод простой итерации", mpi,
                              values=("residuals", "residuals_norm_inf",
                                      "transition_matrix", "transition_vector",
                                      "transition_matrix_norms", "chosen_norm", "iterations"),
                              tables=("iteration_table",),
                              extras=mpi_extras)
            )
        except Exception as e:
            show_error(page, str(e))
            traceback.print_exc()
        page.update()

    return ft.Column(
        [
            section_title("Задание 1", "СЛАУ: метод Гаусса и метод простой итерации"),
            card(
                "Система уравнений",
                ft.Text("Выберите размерность, затем введите коэффициенты у переменных и правую часть.",
                        size=12, color=ft.Colors.ON_SURFACE_VARIANT),
                ft.Row([dimension, eps], spacing=12, wrap=True),
                system_holder,
                action_row(solve_button(solve), variant_button(load_variant)),
            ),
            results,
        ],
        spacing=14,
        scroll=ft.ScrollMode.AUTO,
    )


def page_task2(page: ft.Page) -> ft.Control:
    system = SystemMatrixInput(4)
    system_holder = ft.Container(content=system.build())
    eps = ft.TextField(label="Точность ε", value="", width=160)
    results = results_container()

    def change_dimension(e: ft.ControlEvent) -> None:
        system.resize(int(e.control.value or system.n))
        system_holder.content = system.build()
        results.controls.clear()
        page.update()

    dimension = dimension_dropdown(4, change_dimension, min_n=3, max_n=6)

    def load_variant(_):
        dimension.value = "4"
        system.resize(4)
        system.set_values(
            [[8.0, -4.0, 0.0, 0.0],
             [-2.0, 6.0, -2.0, 0.0],
             [0.0, 2.0, 9.0, -5.0],
             [0.0, 0.0, -2.0, 6.0]],
            [11.0, 5.0, 15.5, 9.5],
        )
        system_holder.content = system.build()
        eps.value = "0.01"
        page.update()

    def solve(_):
        results.controls.clear()
        try:
            A = system.get_A()
            b = system.get_b()
            ev = float((eps.value or "0.01").replace(",", "."))
            progonka = pretty_solve_progonka_matrix(A, b)
            zeidel = pretty_solve_zeidel_matrix(ev, A, b)

            pg_extras: list[ft.Control] = []
            if progonka.get("status") == "ok":
                pg_extras.append(solution_system_view(progonka["solution"], "x", "Ответ"))
            results.controls.append(
                render_result("A) Метод прогонки", progonka,
                              values=("residuals", "residuals_norm_inf", "A_coeff", "B_coeff"),
                              tables=("coefficient_table", "forward_pass_table", "backward_pass_table"),
                              extras=pg_extras)
            )
            zd_extras: list[ft.Control] = []
            if zeidel.get("status") == "ok":
                zd_extras.append(solution_system_view(zeidel["solutions"], "x", "Ответ"))
            results.controls.append(
                render_result("B) Метод Зейделя", zeidel,
                              values=("residuals", "residuals_norm_inf",
                                      "T_matrix", "d_vec",
                                      "transition_matrix_norms", "chosen_norm", "iterations"),
                              tables=("iteration_table", "residual_table"),
                              extras=zd_extras)
            )
        except Exception as e:
            show_error(page, str(e))
            traceback.print_exc()
        page.update()

    return ft.Column(
        [
            section_title("Задание 2", "СЛАУ: метод прогонки и метод Зейделя"),
            card(
                "Система уравнений (трёхдиагональная)",
                ft.Row([dimension, eps], spacing=12, wrap=True),
                system_holder,
                action_row(solve_button(solve), variant_button(load_variant)),
            ),
            results,
        ],
        spacing=14,
        scroll=ft.ScrollMode.AUTO,
    )


def page_task3() -> ft.Control:
    return ft.Container(
        content=ft.Text("Не реализовано", size=28, weight=ft.FontWeight.W_700),
        alignment=ft.Alignment.CENTER,
        expand=True,
    )


def page_task4(page: ft.Page) -> ft.Control:
    expr_field = ft.TextField(label="f(x)", value="", width=360)
    a_field = ft.TextField(label="a", value="", width=120)
    b_field = ft.TextField(label="b", value="", width=120)
    eps_field = ft.TextField(label="Точность ε", value="", width=140)
    plot_area = ft.Column(spacing=14)
    results = results_container()

    def load_variant(_):
        expr_field.value = "x + lg(x) - 0.5"
        a_field.value = "0.1"
        b_field.value = "1.0"
        eps_field.value = "0.01"
        page.update()

    def show_graph(_):
        plot_area.controls.clear()
        try:
            f = make_function_1arg(expr_field.value or "0")
            a_v = float((a_field.value or "0").replace(",", "."))
            b_v = float((b_field.value or "1").replace(",", "."))
            chart, legend = chart_function_only(f, a_v, b_v)
            plot_area.controls.append(
                chart_card(f"График f(x) на [{a_v}, {b_v}]", chart, legend, height=380)
            )
        except Exception as e:
            show_error(page, str(e))
            traceback.print_exc()
        page.update()

    def solve(_):
        results.controls.clear()
        try:
            f = make_function_1arg(expr_field.value or "0")
            a_v = float((a_field.value or "0").replace(",", "."))
            b_v = float((b_field.value or "1").replace(",", "."))
            ev = float((eps_field.value or "0.01").replace(",", "."))
            mpi = pretty_solve_mpi_func(ev, f, a_v, b_v)
            dih = pretty_solve_dihotomy_func(ev, f, a_v, b_v)
            new = pretty_solve_newton_func(ev, f, a_v, b_v)

            results.controls.append(
                render_result("A) Метод простой итерации", mpi,
                              values=("solution", "q", "lambda_val"),
                              tables=("iteration_table",))
            )
            if mpi.get("status") == "ok" and "iteration_table" in mpi:
                df = mpi["iteration_table"]
                results.controls.append(
                    iteration_player(
                        "МПИ — анимация итераций",
                        len(df),
                        lambda k, f=f, a_v=a_v, b_v=b_v, df=df: chart_mpi_step(f, a_v, b_v, df, k),
                        page,
                        x_pad_frac=0.0,
                    )
                )

            results.controls.append(
                render_result("B) Метод половинного деления", dih,
                              values=("solution",), tables=("iteration_table",))
            )
            if dih.get("status") == "ok" and "iteration_table" in dih:
                df = dih["iteration_table"]
                results.controls.append(
                    iteration_player(
                        "Половинное деление — анимация итераций",
                        len(df),
                        lambda k, f=f, df=df: chart_dihotomy_step(f, df, k),
                        page,
                        x_pad_frac=0.0,
                    )
                )

            results.controls.append(
                render_result("C) Метод Ньютона", new,
                              values=("solution", "x_0", "iterations"),
                              tables=("iteration_table",))
            )
            if new.get("status") == "ok" and "iteration_table" in new:
                df = new["iteration_table"]
                results.controls.append(
                    iteration_player(
                        "Ньютон — анимация итераций",
                        len(df),
                        lambda k, f=f, a_v=a_v, b_v=b_v, df=df: chart_newton_step(f, a_v, b_v, df, k),
                        page,
                        x_pad_frac=0.0,
                    )
                )
        except Exception as e:
            show_error(page, str(e))
            traceback.print_exc()
        page.update()

    return ft.Column(
        [
            section_title("Задание 4", "Корень нелинейного уравнения: МПИ, дихотомия, Ньютон"),
            card(
                "Уравнение и параметры",
                ft.Text("Сначала постройте график, чтобы оценить расположение корня и выбрать интервал.",
                        size=12, color=ft.Colors.ON_SURFACE_VARIANT),
                expr_field,
                ft.Row([a_field, b_field, eps_field], spacing=12),
                action_row(
                    secondary_button("Показать график", ft.Icons.SHOW_CHART, show_graph),
                    solve_button(solve),
                    variant_button(load_variant),
                ),
            ),
            plot_area,
            results,
        ],
        spacing=14,
        scroll=ft.ScrollMode.AUTO,
    )


def page_task5(page: ft.Page) -> ft.Control:
    table = XYTableInput(6)
    x_star = ft.TextField(label="x*", value="", width=160)
    results = results_container()

    def load_variant(_):
        table.set_values(
            [1.415, 1.420, 1.425, 1.430, 1.435, 1.440],
            [0.888551, 0.889559, 0.890637, 0.891667, 0.892687, 0.893698],
        )
        x_star.value = "1.4179"
        page.update()

    def solve(_):
        results.controls.clear()
        try:
            df = table.to_dataframe()
            xs_vec, ys_vec = table.get_xy()
            xs = float((x_star.value or "0").replace(",", "."))
            lag = pretty_solve_lagrange_table(df, xs)
            new = pretty_solve_newton_table(df, xs)
            results.controls.append(
                render_result("A) Многочлен Лагранжа", lag,
                              values=("result",), tables=("terms_table",))
            )
            results.controls.append(
                render_result("B) Многочлен Ньютона", new,
                              values=("result", "coefficients", "divided_differences_matrix"),
                              tables=("coefficients_table", "terms_table"))
            )
            lag_val = lag["result"] if lag.get("status") == "ok" else None
            new_val = new["result"] if new.get("status") == "ok" else None
            # формула многочлена Ньютона
            if new.get("status") == "ok" and "coefficients" in new:
                coefs = np.asarray(new["coefficients"]).flatten()
                x_nodes = xs_vec
                terms = [latex_num(coefs[0])]
                for i in range(1, len(coefs)):
                    factors = "".join(rf"(x - {x_nodes[j]:.3f})" for j in range(i))
                    terms.append(latex_signed_term(float(coefs[i]), factors))
                expr = "N(x) = " + "".join(terms)
                results.controls.append(
                    card("Многочлен Ньютона в развёрнутом виде", latex(expr, fontsize=15))
                )
            if lag_val is not None and new_val is not None:
                delta = abs(lag_val - new_val)
                results.controls.append(
                    card(
                        "Сравнение",
                        latex(rf"L(x^{{*}}) = {latex_num(lag_val)}", fontsize=18),
                        latex(rf"N(x^{{*}}) = {latex_num(new_val)}", fontsize=18),
                        latex(rf"\bigl| L - N \bigr| = {latex_num(delta)}", fontsize=18),
                    )
                )
            chart, legend = chart_interpolation(xs_vec, ys_vec, xs, lag_val, new_val)
            results.controls.append(chart_card("График интерполяции", chart, legend, height=400))
        except Exception as e:
            show_error(page, str(e))
            traceback.print_exc()
        page.update()

    return ft.Column(
        [
            section_title("Задание 5", "Интерполяционные многочлены Лагранжа и Ньютона"),
            card(
                "Таблица узлов и точка x*",
                ft.Container(content=table.build(), padding=ft.Padding.symmetric(vertical=4)),
                ft.Row([x_star]),
                action_row(solve_button(solve), variant_button(load_variant)),
            ),
            results,
        ],
        spacing=14,
        scroll=ft.ScrollMode.AUTO,
    )


def page_task6(page: ft.Page) -> ft.Control:
    table = XYTableInput(6)
    results = results_container()

    def load_variant(_):
        table.set_values(
            [1.415, 1.420, 1.425, 1.430, 1.435, 1.440],
            [0.888551, 0.889559, 0.890637, 0.891667, 0.892687, 0.893698],
        )
        page.update()

    def solve(_):
        results.controls.clear()
        try:
            df = table.to_dataframe()
            xs_vec, ys_vec = table.get_xy()
            res = pretty_solve_cubic_splines_table(df)
            results.controls.append(
                render_result(
                    "Кубический сплайн дефекта 1",
                    res,
                    values=("coefficients", "s2_left", "s2_right", "s2_diff"),
                    tables=("h_delta_table", "system_table", "coefficients_table", "spline_table"),
                )
            )
            if res.get("status") == "ok":
                chart, legend = chart_spline(xs_vec, ys_vec, res)
                results.controls.append(chart_card("График сплайна", chart, legend, height=420))
        except Exception as e:
            show_error(page, str(e))
            traceback.print_exc()
        page.update()

    return ft.Column(
        [
            section_title("Задание 6", "Кубические сплайны по таблице задания 5"),
            card(
                "Таблица узлов",
                ft.Container(content=table.build(), padding=ft.Padding.symmetric(vertical=4)),
                action_row(solve_button(solve), variant_button(load_variant)),
            ),
            results,
        ],
        spacing=14,
        scroll=ft.ScrollMode.AUTO,
    )


def page_task7(page: ft.Page) -> ft.Control:
    table = XYTableInput(11)
    results = results_container()

    def load_variant(_):
        table.set_values(
            [0.0, 0.12, 0.19, 0.35, 0.4, 0.45, 0.62, 0.71, 0.84, 0.91, 1.0],
            [-1.0, -1.0, -0.9, -0.5, -0.7, -0.6, -0.3, -0.5, 0.4, 0.8, 1.2],
        )
        page.update()

    def solve(_):
        results.controls.clear()
        try:
            df = table.to_dataframe()
            xs_vec, ys_vec = table.get_xy()
            lin = pretty_solve_linear_mls_table(df)
            quad = pretty_solve_quadratic_mls_table(df)
            results.controls.append(
                render_result("A) Линейный многочлен P₁(x) = a·x + b", lin,
                              values=("a", "b", "residuals_norm_inf"))
            )
            results.controls.append(
                render_result("B) Квадратичный многочлен P₂(x) = a·x² + b·x + c", quad,
                              values=("a", "b", "c", "residuals_norm_inf"))
            )
            a1 = lin.get("a") if lin.get("status") == "ok" else None
            b1 = lin.get("b") if lin.get("status") == "ok" else None
            a2 = quad.get("a") if quad.get("status") == "ok" else None
            b2 = quad.get("b") if quad.get("status") == "ok" else None
            c2 = quad.get("c") if quad.get("status") == "ok" else None
            formula_lines: list[ft.Control] = []
            if a1 is not None and b1 is not None:
                expr1 = "P_1(x) = " + latex_num(a1) + r"\,x" + latex_signed_term(float(b1))
                formula_lines.append(latex(expr1, fontsize=18))
            if a2 is not None and b2 is not None and c2 is not None:
                expr2 = ("P_2(x) = " + latex_num(a2) + r"\,x^2"
                         + latex_signed_term(float(b2), "x")
                         + latex_signed_term(float(c2)))
                formula_lines.append(latex(expr2, fontsize=18))
            if formula_lines:
                results.controls.append(card("Аппроксимирующие многочлены", *formula_lines))
            chart, legend = chart_least_squares(xs_vec, ys_vec, a1, b1, a2, b2, c2)
            results.controls.append(chart_card("График аппроксимации", chart, legend, height=440))
        except Exception as e:
            show_error(page, str(e))
            traceback.print_exc()
        page.update()

    return ft.Column(
        [
            section_title("Задание 7", "Метод наименьших квадратов: линейная и квадратичная аппроксимации"),
            card(
                "Таблица узлов",
                ft.Container(content=table.build(), padding=ft.Padding.symmetric(vertical=4)),
                action_row(solve_button(solve), variant_button(load_variant)),
            ),
            results,
        ],
        spacing=14,
        scroll=ft.ScrollMode.AUTO,
    )


def page_task8(page: ft.Page) -> ft.Control:
    table = XYTableInput(6)
    x_dots = ft.TextField(label="Точки (через запятую)", value="", width=360)
    results = results_container()

    def load_variant(_):
        table.set_values(
            [1.415, 1.420, 1.425, 1.430, 1.435, 1.440],
            [0.888551, 0.889559, 0.890637, 0.891667, 0.892687, 0.893698],
        )
        x_dots.value = "1.4179, 1.420"
        page.update()

    def solve(_):
        results.controls.clear()
        try:
            df = table.to_dataframe()
            xs_vec, ys_vec = table.get_xy()
            dots_raw = [s.strip() for s in (x_dots.value or "").split(",") if s.strip()]
            dots = np.array([float(s.replace(",", ".")) for s in dots_raw], dtype=float)
            first = pretty_solve_first_diff_table(df, dots)
            second = pretty_solve_second_diff_table(df, dots)
            point_labels = [
                "x*" if i == 0 else "x_2" if i == 1 else f"x_{i + 1}"
                for i in range(len(dots))
            ]
            answer_df = local_quadratic_derivatives(
                xs_vec,
                ys_vec,
                dict(zip(point_labels, dots)),
            )
            if not answer_df.empty:
                results.controls.append(
                    card("Ответ в заданных точках", df_to_datatable(answer_df))
                )
            results.controls.append(
                render_result("A) Первая производная", first,
                              values=("result", "uniform_grid"),
                              tables=("diff_table",))
            )
            results.controls.append(
                render_result("B) Вторая производная", second,
                              values=("result", "uniform_grid"),
                              tables=("diff_table",))
            )
        except Exception as e:
            show_error(page, str(e))
            traceback.print_exc()
        page.update()

    return ft.Column(
        [
            section_title("Задание 8", "Первая и вторая производные по табличным данным"),
            card(
                "Таблица узлов и контрольные точки",
                ft.Container(content=table.build(), padding=ft.Padding.symmetric(vertical=4)),
                ft.Row([x_dots]),
                action_row(solve_button(solve), variant_button(load_variant)),
            ),
            results,
        ],
        spacing=14,
        scroll=ft.ScrollMode.AUTO,
    )


def page_task9(page: ft.Page) -> ft.Control:
    expr_field = ft.TextField(label="f(x)", value="", width=360)
    a_field = ft.TextField(label="a", value="", width=120)
    b_field = ft.TextField(label="b", value="", width=120)
    eps_field = ft.TextField(label="Точность ε", value="", width=140)
    results = results_container()

    def load_variant(_):
        expr_field.value = "sqrt(x) * cos(x**2)"
        a_field.value = "0.0"
        b_field.value = "1.0"
        eps_field.value = "0.0001"
        page.update()

    def solve(_):
        results.controls.clear()
        try:
            f = make_function_1arg(expr_field.value or "0")
            a_v = float((a_field.value or "0").replace(",", "."))
            b_v = float((b_field.value or "1").replace(",", "."))
            ev = float((eps_field.value or "1e-4").replace(",", "."))
            res = pretty_solve_simpson_integration_func(f, a_v, b_v, ev)
            results.controls.append(
                render_result("Интеграл методом Симпсона", res,
                              values=("result", "a", "b", "eps", "iterations", "n"),
                              tables=("iteration_table",))
            )
            chart, legend = chart_simpson(f, a_v, b_v)
            results.controls.append(chart_card("Площадь под кривой", chart, legend, height=400))
        except Exception as e:
            show_error(page, str(e))
            traceback.print_exc()
        page.update()

    return ft.Column(
        [
            section_title("Задание 9", "Определённый интеграл методом Симпсона"),
            card(
                "Подынтегральная функция и пределы",
                expr_field,
                ft.Row([a_field, b_field, eps_field], spacing=12),
                action_row(solve_button(solve), variant_button(load_variant)),
            ),
            results,
        ],
        spacing=14,
        scroll=ft.ScrollMode.AUTO,
    )


def page_task10(page: ft.Page) -> ft.Control:
    expr_field = ft.TextField(label="f(x, y) = y'", value="", width=460)
    h_field = ft.TextField(label="Шаг h", value="", width=120)
    x0_field = ft.TextField(label="x₀", value="", width=120)
    y0_field = ft.TextField(label="y₀", value="", width=120)
    xstar_field = ft.TextField(label="x*", value="", width=120)
    results = results_container()

    def load_variant(_):
        expr_field.value = "1 - sin(x + y) + 0.5*y/(x + 2)"
        h_field.value = "0.1"
        x0_field.value = "0.0"
        y0_field.value = "0.0"
        xstar_field.value = "1.0"
        page.update()

    def solve(_):
        results.controls.clear()
        try:
            f = make_function_2arg(expr_field.value or "0")
            h_v = float((h_field.value or "0.1").replace(",", "."))
            x0_v = float((x0_field.value or "0").replace(",", "."))
            y0_v = float((y0_field.value or "0").replace(",", "."))
            xs_v = float((xstar_field.value or "1").replace(",", "."))
            rk = pretty_solve_runge_cutta_func(f, h_v, x0_v, y0_v, xs_v)
            ad = pretty_solve_adams_func(f, h_v, x0_v, y0_v, xs_v)
            rk_view = dict(rk)
            ad_view = dict(ad)
            if rk.get("status") == "ok":
                rk_view["solution_table"] = solution_xy_table(rk)
            if ad.get("status") == "ok":
                ad_view["solution_table"] = solution_xy_table(ad)
            results.controls.append(
                render_result("A) Метод Рунге-Кутты 4-го порядка", rk_view,
                              values=("result", "h", "x_0", "y_0"),
                              tables=("solution_table", "step_table", "interpolation_table"))
            )
            results.controls.append(
                render_result("B) Метод Адамса", ad_view,
                              values=("result", "h", "x_0", "y_0"),
                              tables=("solution_table", "step_table", "interpolation_table"))
            )
            chart, legend = chart_cauchy(rk, ad)
            results.controls.append(chart_card("График решения y(x)", chart, legend, height=440))
        except Exception as e:
            show_error(page, str(e))
            traceback.print_exc()
        page.update()

    return ft.Column(
        [
            section_title("Задание 10", "Задача Коши: методы Рунге-Кутты и Адамса"),
            card(
                "ОДУ и начальные данные",
                expr_field,
                ft.Row([h_field, x0_field, y0_field, xstar_field], spacing=12),
                action_row(solve_button(solve), variant_button(load_variant)),
            ),
            results,
        ],
        spacing=14,
        scroll=ft.ScrollMode.AUTO,
    )


def page_task11(page: ft.Page) -> ft.Control:
    p_expr = ft.TextField(label="p(x)", value="", width=200)
    q_expr = ft.TextField(label="q(x)", value="", width=200)
    f_expr = ft.TextField(label="f(x)", value="", width=200)
    a_in = ft.TextField(label="a", value="", width=120)
    b_in = ft.TextField(label="b", value="", width=120)
    h_in = ft.TextField(label="h", value="", width=120)
    alpha = VectorInput(3, width=110)
    beta = VectorInput(3, width=110)
    results = results_container()

    def load_variant(_):
        p_expr.value = "-1.0"
        q_expr.value = "2.0 / x"
        f_expr.value = "x + 0.4"
        a_in.value = "1.0"
        b_in.value = "1.4"
        h_in.value = "0.1"
        alpha.set_values([1.0, -0.5, 2.0])
        beta.set_values([0.0, 1.0, 4.0])
        page.update()

    def solve(_):
        results.controls.clear()
        try:
            p = make_function_1arg(p_expr.value or "0")
            q = make_function_1arg(q_expr.value or "0")
            f = make_function_1arg(f_expr.value or "0")
            h_v = float((h_in.value or "0.1").replace(",", "."))
            a_v = float((a_in.value or "0").replace(",", "."))
            b_v = float((b_in.value or "1").replace(",", "."))
            res = pretty_solve_progonka_func(h_v, p, q, f, a_v, b_v, alpha.get(), beta.get())
            results.controls.append(
                render_result(
                    "Метод прогонки для краевой задачи",
                    res,
                    values=("x", "y", "h", "residuals", "residuals_norm_inf"),
                    tables=("grid_table", "boundary_table", "equation_table", "forward_pass_table"),
                )
            )
            if res.get("status") == "ok":
                chart, legend = chart_boundary(res)
                results.controls.append(chart_card("График решения y(x)", chart, legend, height=400))
        except Exception as e:
            show_error(page, str(e))
            traceback.print_exc()
        page.update()

    return ft.Column(
        [
            section_title("Задание 11", "Краевая задача для ОДУ методом прогонки"),
            card(
                "Уравнение y'' + p(x)·y' + q(x)·y = f(x)",
                ft.Row([p_expr, q_expr, f_expr], spacing=12, wrap=True),
                ft.Row([a_in, b_in, h_in], spacing=12),
                ft.Text("Граничные условия [α₀·y + α₁·y' = α₂] и [β₀·y + β₁·y' = β₂]:"),
                ft.Row([ft.Container(ft.Text("α", weight=ft.FontWeight.W_600), width=24), alpha.build()], spacing=4),
                ft.Row([ft.Container(ft.Text("β", weight=ft.FontWeight.W_600), width=24), beta.build()], spacing=4),
                action_row(solve_button(solve), variant_button(load_variant)),
            ),
            results,
        ],
        spacing=14,
        scroll=ft.ScrollMode.AUTO,
    )


# ===== Приложение =====

def main(page: ft.Page) -> None:
    page.title = "Численные методы"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.theme = ft.Theme(color_scheme_seed=ft.Colors.INDIGO, use_material3=True)
    page.padding = 0
    page.window.width = 1320
    page.window.height = 860
    page.window.min_width = 980
    page.window.min_height = 600

    content_area = ft.Container(expand=True, padding=ft.Padding.all(20))

    def builder(index: int) -> ft.Control:
        if index == 0:
            return page_home()
        if index == 1:
            return page_task1(page)
        if index == 2:
            return page_task2(page)
        if index == 3:
            return page_task3()
        if index == 4:
            return page_task4(page)
        if index == 5:
            return page_task5(page)
        if index == 6:
            return page_task6(page)
        if index == 7:
            return page_task7(page)
        if index == 8:
            return page_task8(page)
        if index == 9:
            return page_task9(page)
        if index == 10:
            return page_task10(page)
        if index == 11:
            return page_task11(page)
        return page_home()

    def on_change(e: ft.ControlEvent) -> None:
        idx = int(e.control.selected_index)
        content_area.content = builder(idx)
        page.update()

    def dest(icon: str, label: str) -> ft.NavigationRailDestination:
        return ft.NavigationRailDestination(icon=icon, label=label)

    def number_dest(number: int) -> ft.NavigationRailDestination:
        return ft.NavigationRailDestination(
            icon=rail_number_icon(number),
            selected_icon=rail_number_icon(number, selected=True),
            label=f"№{number}",
        )

    rail = ft.NavigationRail(
        selected_index=0,
        label_type=ft.NavigationRailLabelType.ALL,
        min_width=84,
        min_extended_width=180,
        leading=ft.Container(
            content=ft.Column(
                [
                    ft.Icon(ft.Icons.FUNCTIONS, size=32, color=ft.Colors.INDIGO),
                    ft.Text("ЧМ", size=12, weight=ft.FontWeight.BOLD),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=2,
            ),
            padding=ft.Padding.symmetric(vertical=12),
        ),
        group_alignment=-0.95,
        destinations=[
            dest(ft.Icons.HOME, "Главная"),
            *[number_dest(i) for i in range(1, 12)],
        ],
        on_change=on_change,
    )

    content_area.content = builder(0)

    page.add(
        ft.Row(
            [rail, ft.VerticalDivider(width=1), content_area],
            expand=True,
            spacing=0,
        )
    )


if __name__ == "__main__":
    ft.run(main)
