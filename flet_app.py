"""Flet интерфейс для лабораторной работы по численным методам."""

from __future__ import annotations

import traceback
from typing import Any, Callable

import flet as ft
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
    "step_table": "Шаги метода",
    "interpolation_table": "Интерполяция",
    "diff_table": "Разностная таблица",
}


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


# ===== Компоненты =====

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


def df_to_datatable(df: pd.DataFrame) -> ft.Control:
    if df.empty:
        return ft.Text("(пусто)", italic=True, color=ft.Colors.ON_SURFACE_VARIANT)
    columns = [ft.DataColumn(ft.Text(str(c))) for c in df.columns]
    rows: list[ft.DataRow] = []
    for _, row in df.iterrows():
        cells: list[ft.DataCell] = []
        for value in row.tolist():
            if isinstance(value, (float, np.floating, int, np.integer)):
                text = fmt(value)
            elif value is None or (isinstance(value, float) and pd.isna(value)):
                text = "-"
            else:
                text = str(value)
            cells.append(ft.DataCell(ft.Text(text, size=12)))
        rows.append(ft.DataRow(cells=cells))
    return ft.Container(
        content=ft.DataTable(
            columns=columns,
            rows=rows,
            heading_row_height=36,
            data_row_min_height=28,
            data_row_max_height=40,
        ),
        padding=ft.Padding.symmetric(vertical=4),
    )


def value_block(label: str, value: Any) -> ft.Control:
    title = LABELS.get(label, label)
    if isinstance(value, pd.DataFrame):
        return ft.Column([ft.Text(title, weight=ft.FontWeight.W_600), df_to_datatable(value)], spacing=6)
    if isinstance(value, np.ndarray):
        return ft.Column(
            [
                ft.Text(title, weight=ft.FontWeight.W_600),
                ft.Container(
                    content=ft.Text(fmt_array(value), font_family="Consolas", selectable=True),
                    bgcolor=ft.Colors.SURFACE_CONTAINER,
                    padding=ft.Padding.all(8),
                    border_radius=6,
                ),
            ],
            spacing=4,
        )
    if isinstance(value, (list, tuple)) and value and all(isinstance(v, np.ndarray) for v in value):
        items: list[ft.Control] = [ft.Text(title, weight=ft.FontWeight.W_600)]
        for i, arr in enumerate(value, 1):
            items.append(
                ft.Container(
                    content=ft.Text(f"[{i}] {fmt_array(arr)}", font_family="Consolas", selectable=True),
                    bgcolor=ft.Colors.SURFACE_CONTAINER,
                    padding=ft.Padding.all(8),
                    border_radius=6,
                )
            )
        return ft.Column(items, spacing=4)
    return ft.Row(
        [ft.Text(f"{title}:", weight=ft.FontWeight.W_600), ft.Text(fmt(value), selectable=True)],
        spacing=8,
    )


def render_result(
    title: str,
    result: dict[str, Any],
    values: tuple[str, ...] = (),
    tables: tuple[str, ...] = (),
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
    return card(title, *blocks)


# ===== Ввод матриц / векторов / таблиц =====

class MatrixInput:
    def __init__(self, rows: int, cols: int, width: int = 90):
        self.rows = rows
        self.cols = cols
        self.fields: list[list[ft.TextField]] = []
        for _ in range(rows):
            row: list[ft.TextField] = []
            for _ in range(cols):
                row.append(
                    ft.TextField(
                        value="",
                        text_align=ft.TextAlign.RIGHT,
                        width=width,
                        height=42,
                        text_size=13,
                        content_padding=ft.Padding.symmetric(horizontal=6, vertical=4),
                    )
                )
            self.fields.append(row)

    def build(self) -> ft.Control:
        return ft.Column([ft.Row(row, spacing=4) for row in self.fields], spacing=4)

    def set_values(self, matrix: list[list[float]]) -> None:
        for i, row in enumerate(matrix):
            for j, v in enumerate(row):
                self.fields[i][j].value = f"{v:.4f}"

    def get(self) -> np.ndarray:
        data: list[list[float]] = []
        for row in self.fields:
            data.append([float((cell.value or "0").replace(",", ".")) for cell in row])
        return np.array(data, dtype=float)


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


# ===== Утилиты =====

def show_error(page: ft.Page, message: str) -> None:
    page.open(ft.SnackBar(content=ft.Text(f"Ошибка: {message}"), bgcolor=ft.Colors.RED_700))


def results_container() -> ft.Column:
    return ft.Column(spacing=14, scroll=ft.ScrollMode.AUTO)


def action_row(*buttons: ft.Control) -> ft.Row:
    return ft.Row(list(buttons), spacing=10)


def variant_button(on_click) -> ft.Control:
    return ft.OutlinedButton("Вариант 6", icon=ft.Icons.DOWNLOAD, on_click=on_click)


def solve_button(on_click) -> ft.Control:
    return ft.FilledButton("Решить", icon=ft.Icons.PLAY_ARROW, on_click=on_click)


# ===== Страницы =====

def page_home() -> ft.Control:
    return ft.Column(
        [
            ft.Text("Численные методы", size=28, weight=ft.FontWeight.BOLD),
        ],
        spacing=14,
        scroll=ft.ScrollMode.AUTO,
    )


def page_task1(page: ft.Page) -> ft.Control:
    A = MatrixInput(3, 3)
    b = VectorInput(3)
    eps = ft.TextField(label="Точность ε", value="", width=160)
    results = results_container()

    def load_variant(_):
        A.set_values([[1.61, 0.71, -0.05], [-1.03, -2.05, 0.87], [2.50, -3.12, -6.03]])
        b.set_values([0.44, -1.16, -7.50])
        eps.value = "0.01"
        page.update()

    def solve(_):
        results.controls.clear()
        try:
            Am, bm = A.get(), b.get()
            ev = float((eps.value or "0.01").replace(",", "."))
            gauss = pretty_solve_gauss_matrix(Am, bm)
            mpi = pretty_solve_mpi_slay(ev, Am, bm)
            results.controls.append(
                render_result("A) Метод Гаусса", gauss,
                              values=("solution", "residual", "residual_norm_inf", "matrix_after_forward_pass"))
            )
            results.controls.append(
                render_result("B) Метод простой итерации", mpi,
                              values=("solutions", "residuals", "residuals_norm_inf",
                                      "transition_matrix", "transition_vector",
                                      "transition_matrix_norms", "chosen_norm", "iterations"),
                              tables=("iteration_table",))
            )
        except Exception as e:
            show_error(page, str(e))
            traceback.print_exc()
        page.update()

    return ft.Column(
        [
            section_title("Задание 1", "СЛАУ: метод Гаусса и метод простой итерации"),
            card(
                "Исходная система",
                ft.Text("Матрица коэффициентов A:"),
                A.build(),
                ft.Text("Вектор правых частей b:"),
                b.build(),
                ft.Row([eps], spacing=12),
                action_row(solve_button(solve), variant_button(load_variant)),
            ),
            results,
        ],
        spacing=14,
        scroll=ft.ScrollMode.AUTO,
    )


def page_task2(page: ft.Page) -> ft.Control:
    A = MatrixInput(4, 4)
    b = VectorInput(4)
    eps = ft.TextField(label="Точность ε", value="", width=160)
    results = results_container()

    def load_variant(_):
        A.set_values([
            [8.0, -4.0, 0.0, 0.0],
            [-2.0, 6.0, -2.0, 0.0],
            [0.0, 2.0, 9.0, -5.0],
            [0.0, 0.0, -2.0, 6.0],
        ])
        b.set_values([11.0, 5.0, 15.5, 9.5])
        eps.value = "0.01"
        page.update()

    def solve(_):
        results.controls.clear()
        try:
            Am, bm = A.get(), b.get()
            ev = float((eps.value or "0.01").replace(",", "."))
            progonka = pretty_solve_progonka_matrix(Am, bm)
            zeidel = pretty_solve_zeidel_matrix(ev, Am, bm)
            results.controls.append(
                render_result("A) Метод прогонки", progonka,
                              values=("solution", "residuals", "residuals_norm_inf", "A_coeff", "B_coeff"),
                              tables=("coefficient_table", "forward_pass_table", "backward_pass_table"))
            )
            results.controls.append(
                render_result("B) Метод Зейделя", zeidel,
                              values=("solutions", "residuals", "residuals_norm_inf",
                                      "T_matrix", "d_vec",
                                      "transition_matrix_norms", "chosen_norm", "iterations"),
                              tables=("iteration_table", "residual_table"))
            )
        except Exception as e:
            show_error(page, str(e))
            traceback.print_exc()
        page.update()

    return ft.Column(
        [
            section_title("Задание 2", "СЛАУ: метод прогонки и метод Зейделя"),
            card(
                "Исходная система",
                ft.Text("Матрица A (трёхдиагональная):"),
                A.build(),
                ft.Text("Вектор b:"),
                b.build(),
                ft.Row([eps]),
                action_row(solve_button(solve), variant_button(load_variant)),
            ),
            results,
        ],
        spacing=14,
        scroll=ft.ScrollMode.AUTO,
    )


def page_task3() -> ft.Control:
    return ft.Column([section_title("Задание 3", "Метод вращения")], spacing=14)


def page_task4(page: ft.Page) -> ft.Control:
    expr = ft.TextField(label="f(x)", value="", width=360)
    a = ft.TextField(label="a", value="", width=120)
    b = ft.TextField(label="b", value="", width=120)
    eps = ft.TextField(label="Точность ε", value="", width=140)
    results = results_container()

    def load_variant(_):
        expr.value = "x + lg(x) - 0.5"
        a.value = "0.1"
        b.value = "1.0"
        eps.value = "0.01"
        page.update()

    def solve(_):
        results.controls.clear()
        try:
            f = make_function_1arg(expr.value or "0")
            a_v = float((a.value or "0").replace(",", "."))
            b_v = float((b.value or "1").replace(",", "."))
            ev = float((eps.value or "0.01").replace(",", "."))
            mpi = pretty_solve_mpi_func(ev, f, a_v, b_v)
            dih = pretty_solve_dihotomy_func(ev, f, a_v, b_v)
            new = pretty_solve_newton_func(ev, f, a_v, b_v)
            results.controls.append(
                render_result("A) Метод простой итерации", mpi,
                              values=("solution", "q", "lambda_val"),
                              tables=("iteration_table",))
            )
            results.controls.append(
                render_result("B) Метод половинного деления", dih,
                              values=("solution",), tables=("iteration_table",))
            )
            results.controls.append(
                render_result("C) Метод Ньютона", new,
                              values=("solution", "x_0", "iterations"),
                              tables=("iteration_table",))
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
                ft.Text("Формула вида Python-выражения. Доступны sin, cos, exp, log, lg, sqrt, **, и т.д."),
                expr,
                ft.Row([a, b, eps], spacing=12),
                action_row(solve_button(solve), variant_button(load_variant)),
            ),
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
            if lag.get("status") == "ok" and new.get("status") == "ok":
                delta = abs(lag["result"] - new["result"])
                results.controls.append(
                    card(
                        "Сравнение",
                        ft.Text(f"L(x*) = {fmt(lag['result'])}"),
                        ft.Text(f"N(x*) = {fmt(new['result'])}"),
                        ft.Text(f"|L - N| = {fmt(delta)}", weight=ft.FontWeight.W_600),
                    )
                )
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
            res = pretty_solve_cubic_splines_table(df)
            results.controls.append(
                render_result(
                    "Кубический сплайн дефекта 1",
                    res,
                    values=("coefficients", "s2_left", "s2_right", "s2_diff"),
                    tables=("h_delta_table", "system_table", "coefficients_table", "spline_table"),
                )
            )
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
            dots_raw = [s.strip() for s in (x_dots.value or "").split(",") if s.strip()]
            dots = np.array([float(s.replace(",", ".")) for s in dots_raw], dtype=float)
            first = pretty_solve_first_diff_table(df, dots)
            second = pretty_solve_second_diff_table(df, dots)
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
    expr = ft.TextField(label="f(x)", value="", width=360)
    a = ft.TextField(label="a", value="", width=120)
    b = ft.TextField(label="b", value="", width=120)
    eps = ft.TextField(label="Точность ε", value="", width=140)
    results = results_container()

    def load_variant(_):
        expr.value = "sqrt(x) * cos(x**2)"
        a.value = "0.0"
        b.value = "1.0"
        eps.value = "0.0001"
        page.update()

    def solve(_):
        results.controls.clear()
        try:
            f = make_function_1arg(expr.value or "0")
            a_v = float((a.value or "0").replace(",", "."))
            b_v = float((b.value or "1").replace(",", "."))
            ev = float((eps.value or "1e-4").replace(",", "."))
            res = pretty_solve_simpson_integration_func(f, a_v, b_v, ev)
            results.controls.append(
                render_result("Интеграл методом Симпсона", res,
                              values=("result", "a", "b", "eps", "iterations", "n"),
                              tables=("iteration_table",))
            )
        except Exception as e:
            show_error(page, str(e))
            traceback.print_exc()
        page.update()

    return ft.Column(
        [
            section_title("Задание 9", "Определённый интеграл методом Симпсона"),
            card(
                "Подынтегральная функция и пределы",
                expr,
                ft.Row([a, b, eps], spacing=12),
                action_row(solve_button(solve), variant_button(load_variant)),
            ),
            results,
        ],
        spacing=14,
        scroll=ft.ScrollMode.AUTO,
    )


def page_task10(page: ft.Page) -> ft.Control:
    expr = ft.TextField(label="f(x, y) = y'", value="", width=460)
    h = ft.TextField(label="Шаг h", value="", width=120)
    x0 = ft.TextField(label="x₀", value="", width=120)
    y0 = ft.TextField(label="y₀", value="", width=120)
    xstar = ft.TextField(label="x*", value="", width=120)
    results = results_container()

    def load_variant(_):
        expr.value = "1 - sin(x + y) + 0.5*y/(x + 2)"
        h.value = "0.1"
        x0.value = "0.0"
        y0.value = "0.0"
        xstar.value = "1.0"
        page.update()

    def solve(_):
        results.controls.clear()
        try:
            f = make_function_2arg(expr.value or "0")
            h_v = float((h.value or "0.1").replace(",", "."))
            x0_v = float((x0.value or "0").replace(",", "."))
            y0_v = float((y0.value or "0").replace(",", "."))
            xs_v = float((xstar.value or "1").replace(",", "."))
            rk = pretty_solve_runge_cutta_func(f, h_v, x0_v, y0_v, xs_v)
            ad = pretty_solve_adams_func(f, h_v, x0_v, y0_v, xs_v)
            results.controls.append(
                render_result("A) Метод Рунге-Кутты 4-го порядка", rk,
                              values=("result", "h", "x_0", "y_0", "x", "y"),
                              tables=("step_table", "interpolation_table"))
            )
            results.controls.append(
                render_result("B) Метод Адамса", ad,
                              values=("result", "h", "x_0", "y_0", "x", "y"),
                              tables=("step_table", "interpolation_table"))
            )
        except Exception as e:
            show_error(page, str(e))
            traceback.print_exc()
        page.update()

    return ft.Column(
        [
            section_title("Задание 10", "Задача Коши: методы Рунге-Кутты и Адамса"),
            card(
                "ОДУ и начальные данные",
                expr,
                ft.Row([h, x0, y0, xstar], spacing=12),
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
    page.window.width = 1280
    page.window.height = 820
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
            dest(ft.Icons.LOOKS_ONE, "№1"),
            dest(ft.Icons.LOOKS_TWO, "№2"),
            dest(ft.Icons.LOOKS_3, "№3"),
            dest(ft.Icons.LOOKS_4, "№4"),
            dest(ft.Icons.LOOKS_5, "№5"),
            dest(ft.Icons.LOOKS_6, "№6"),
            dest(ft.Icons.FILTER_7, "№7"),
            dest(ft.Icons.FILTER_8, "№8"),
            dest(ft.Icons.FILTER_9, "№9"),
            dest(ft.Icons.FILTER_9_PLUS, "№10"),
            dest(ft.Icons.NUMBERS, "№11"),
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
