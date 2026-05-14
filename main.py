from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pandas as pd

from interface.console import (
    line,
    print_result,
    print_table,
    print_task_brief,
    print_text_block,
    print_value,
    section,
    subsection,
)
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


EPS = 0.01
X_STAR = 1.4179


def make_xy_table(x_values: list[float], y_values: list[float]) -> pd.DataFrame:
    return pd.DataFrame([["x", *x_values], ["y", *y_values]])


def readable_xy_table(x_values: list[float], y_values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"x_i": x_values, "y_i": y_values})


def array_inline(value: Any) -> str:
    return np.array2string(
        np.asarray(value),
        precision=6,
        suppress_small=True,
        separator=", ",
        floatmode="fixed",
        max_line_width=500,
    )


def number_inline(value: Any) -> str:
    return f"{float(value):.6f}"


def signed_term(value: float, body: str = "") -> str:
    sign = "+" if value >= 0 else "-"
    suffix = f"*{body}" if body else ""
    return f" {sign} {abs(value):.6f}{suffix}"


def linear_polynomial(a: float, b: float) -> str:
    return f"{a:.6f}*x{signed_term(b)}"


def quadratic_polynomial(a: float, b: float, c: float) -> str:
    return f"{a:.6f}*x^2{signed_term(b, 'x')}{signed_term(c)}"


def lagrange_polynomial(x: list[float], y: list[float]) -> str:
    terms: list[str] = []
    for j, y_j in enumerate(y):
        factors = [
            f"(x - {x_i:.3f})/({x[j]:.3f} - {x_i:.3f})"
            for i, x_i in enumerate(x)
            if i != j
        ]
        terms.append(f"{y_j:.6f} * " + " * ".join(factors))
    return "L(x) = " + "\n     + ".join(terms)


def newton_polynomial(x: list[float], coefficients: np.ndarray) -> str:
    terms = [f"{coefficients[0]:.6f}"]
    for i in range(1, len(coefficients)):
        factors = "".join(f"(x - {x[j]:.3f})" for j in range(i))
        terms.append(f"{signed_term(float(coefficients[i])).strip()}*{factors}")
    return "P(x) = " + " ".join(terms).replace("+ -", "- ")


def result_value(result: dict[str, Any], key: str) -> str:
    if result.get("status") != "ok":
        return f"ошибка: {result.get('message', 'нет результата')}"

    value = result[key]
    if isinstance(value, (np.ndarray, list, tuple)):
        return array_inline(value)
    if isinstance(value, (float, np.floating)):
        return number_inline(value)
    return str(value)


def result_norm(result: dict[str, Any], key: str) -> str:
    if result.get("status") != "ok" or key not in result:
        return "-"
    return number_inline(result[key])


def task_brief(
    task: str,
    condition: str,
    restrictions: Iterable[str],
    answers: Iterable[str],
) -> None:
    print_task_brief(task, condition, restrictions, answers)


def details() -> None:
    subsection("Подробные расчеты и таблицы")


def show_system(A: np.ndarray, b: np.ndarray) -> None:
    print_value("A", A)
    print_value("b", b)


def local_quadratic_derivatives(
    x_values: list[float], y_values: list[float], points: dict[str, float]
) -> pd.DataFrame:
    x = np.array(x_values, dtype=float)
    y = np.array(y_values, dtype=float)
    rows: list[dict[str, float | str]] = []

    for label, point in points.items():
        nearest_right = int(np.searchsorted(x, point))
        start = max(0, min(nearest_right - 1, len(x) - 3))
        selected = slice(start, start + 3)
        coefficients = np.polyfit(x[selected], y[selected], deg=2)
        first_derivative = np.polyval(np.polyder(coefficients, 1), point)
        second_derivative = np.polyval(np.polyder(coefficients, 2), point)

        rows.append(
            {
                "point": label,
                "x": point,
                "nodes": ", ".join(f"{value:.3f}" for value in x[selected]),
                "y'": first_derivative,
                "y''": second_derivative,
            }
        )

    return pd.DataFrame(rows)


def task_1() -> None:
    A = np.array(
        [
            [1.61, 0.71, -0.05],
            [-1.03, -2.05, 0.87],
            [2.50, -3.12, -6.03],
        ],
        dtype=float,
    )
    b = np.array([0.44, -1.16, -7.50], dtype=float)

    gauss_result = pretty_solve_gauss_matrix(A, b)
    mpi_result = pretty_solve_mpi_slay(EPS, A, b)

    section("Задание 1. СЛАУ: метод Гаусса и метод простой итерации")
    task_brief(
        task="Решить СЛАУ двумя методами.",
        condition=(
            "1.61*x1 + 0.71*x2 - 0.05*x3 = 0.44; "
            "-1.03*x1 - 2.05*x2 + 0.87*x3 = -1.16; "
            "2.50*x1 - 3.12*x2 - 6.03*x3 = -7.50."
        ),
        restrictions=(
            f"Точность для метода простой итерации: eps = {EPS}.",
            "Методы: A) Гаусс, B) простая итерация.",
        ),
        answers=(
            f"Гаусс: x = {result_value(gauss_result, 'solution')}, "
            f"||r||inf = {result_norm(gauss_result, 'residual_norm_inf')}.",
            f"Простая итерация: x = {result_value(mpi_result, 'solutions')}, "
            f"итераций = {mpi_result.get('iterations', '-')}, "
            f"||r||inf = {result_norm(mpi_result, 'residuals_norm_inf')}.",
        ),
    )

    details()
    show_system(A, b)
    print_result(
        "A) Метод Гаусса",
        gauss_result,
        values=("solution", "residual", "residual_norm_inf", "matrix_after_forward_pass"),
    )
    print_result(
        "B) Метод простой итерации",
        mpi_result,
        values=(
            "solutions",
            "residuals",
            "residuals_norm_inf",
            "transition_matrix",
            "transition_vector",
            "transition_matrix_norms",
            "chosen_norm",
            "iterations",
        ),
        tables=("iteration_table",),
    )


def task_2() -> None:
    A = np.array(
        [
            [8.0, -4.0, 0.0, 0.0],
            [-2.0, 6.0, -2.0, 0.0],
            [0.0, 2.0, 9.0, -5.0],
            [0.0, 0.0, -2.0, 6.0],
        ],
        dtype=float,
    )
    b = np.array([11.0, 5.0, 15.5, 9.5], dtype=float)

    progonka_result = pretty_solve_progonka_matrix(A, b)
    zeidel_result = pretty_solve_zeidel_matrix(EPS, A, b)

    section("Задание 2. СЛАУ: метод прогонки и метод Зейделя")
    task_brief(
        task="Решить СЛАУ двумя методами.",
        condition=(
            "8*x1 - 4*x2 = 11.0; "
            "-2*x1 + 6*x2 - 2*x3 = 5.0; "
            "2*x2 + 9*x3 - 5*x4 = 15.5; "
            "-2*x3 + 6*x4 = 9.5."
        ),
        restrictions=(
            f"Точность для метода Зейделя: eps = {EPS}.",
            "Методы: A) прогонка для трехдиагональной матрицы, B) Зейдель.",
        ),
        answers=(
            f"Прогонка: x = {result_value(progonka_result, 'solution')}, "
            f"||r||inf = {result_norm(progonka_result, 'residuals_norm_inf')}.",
            f"Зейдель: x = {result_value(zeidel_result, 'solutions')}, "
            f"итераций = {zeidel_result.get('iterations', '-')}, "
            f"||r||inf = {result_norm(zeidel_result, 'residuals_norm_inf')}.",
        ),
    )

    details()
    show_system(A, b)
    print_result(
        "A) Метод прогонки",
        progonka_result,
        values=("solution", "residuals", "residuals_norm_inf", "A_coeff", "B_coeff"),
        tables=("coefficient_table", "forward_pass_table", "backward_pass_table"),
    )
    print_result(
        "B) Метод Зейделя",
        zeidel_result,
        values=(
            "solutions",
            "residuals",
            "residuals_norm_inf",
            "T_matrix",
            "d_vec",
            "transition_matrix_norms",
            "chosen_norm",
            "iterations",
        ),
        tables=("iteration_table", "residual_table"),
    )


def task_3() -> None:
    section("Задание 3. Метод вращения")
    task_brief(
        task="Найти собственные значения и собственные векторы симметричной матрицы методом вращения.",
        condition="A = [[50 + 3n, 10 - n, 3], [10 - n, 20 + 2n, 10 - n], [3, 10 - n, 90 - n]].",
        restrictions=(
            f"Точность: eps = {EPS}.",
            "n - номер варианта.",
            "По текущему запросу пользователя метод вращения не добавляется и не запускается.",
        ),
        answers=("Расчет пропущен: пользователь попросил не добавлять вращение.",),
    )


def task_4() -> None:
    def f(x: float) -> float:
        return x + math.log10(x) - 0.5

    a, b = 0.1, 1.0
    mpi_result = pretty_solve_mpi_func(EPS, f, a, b)
    dihotomy_result = pretty_solve_dihotomy_func(EPS, f, a, b)
    newton_result = pretty_solve_newton_func(EPS, f, a, b)

    section("Задание 4. Корень уравнения x + lg(x) = 0.5")
    task_brief(
        task="Уточнить один корень нелинейного уравнения тремя методами.",
        condition="f(x) = x + lg(x) - 0.5 = 0, где lg(x) = log10(x).",
        restrictions=(
            f"Точность: eps = {EPS}.",
            f"Рабочий отрезок: [{a}, {b}].",
            "Методы: A) простая итерация, B) половинное деление, C) Ньютон.",
        ),
        answers=(
            f"Простая итерация: x = {result_value(mpi_result, 'solution')}.",
            f"Половинное деление: x = {result_value(dihotomy_result, 'solution')}.",
            f"Ньютон: x = {result_value(newton_result, 'solution')}, "
            f"итераций = {newton_result.get('iterations', '-')}.",
        ),
    )

    details()
    print_result(
        "A) Метод простой итерации",
        mpi_result,
        values=("solution", "q", "lambda_val"),
        tables=("iteration_table",),
    )
    print_result(
        "B) Метод половинного деления",
        dihotomy_result,
        values=("solution",),
        tables=("iteration_table",),
    )
    print_result(
        "C) Метод Ньютона",
        newton_result,
        values=("solution", "x_0", "iterations"),
        tables=("iteration_table",),
    )


def task_5_and_6_tables() -> tuple[list[float], list[float], pd.DataFrame]:
    x = [1.415, 1.420, 1.425, 1.430, 1.435, 1.440]
    y = [0.888551, 0.889559, 0.890637, 0.891667, 0.892687, 0.893698]
    return x, y, make_xy_table(x, y)


def task_5() -> None:
    x, y, table = task_5_and_6_tables()
    lagrange_result = pretty_solve_lagrange_table(table, X_STAR)
    newton_result = pretty_solve_newton_table(table, X_STAR)
    delta = (
        abs(lagrange_result["result"] - newton_result["result"])
        if lagrange_result["status"] == "ok" and newton_result["status"] == "ok"
        else None
    )

    section("Задание 5. Интерполяционные многочлены Лагранжа и Ньютона")
    task_brief(
        task="Выписать интерполяционные многочлены Лагранжа и Ньютона и найти значение в x*.",
        condition="Таблица узлов: x = [1.415, 1.420, 1.425, 1.430, 1.435, 1.440], y задана ниже.",
        restrictions=(
            f"x* = {X_STAR}.",
            "Используются все 6 узлов таблицы.",
        ),
        answers=(
            f"Лагранж: P(x*) = {result_value(lagrange_result, 'result')}.",
            f"Ньютон: P(x*) = {result_value(newton_result, 'result')}.",
            f"|L(x*) - N(x*)| = {number_inline(delta) if delta is not None else '-'}.",
        ),
    )

    details()
    print_table("Узлы", readable_xy_table(x, y))
    print_result(
        "A) Многочлен Лагранжа",
        lagrange_result,
        values=("result",),
        tables=("terms_table",),
    )
    print_result(
        "B) Многочлен Ньютона",
        newton_result,
        values=("result", "coefficients", "divided_differences_matrix"),
        tables=("coefficients_table", "terms_table"),
    )

    if delta is not None:
        comparison = pd.DataFrame(
            [
                {
                    "Lagrange(x*)": lagrange_result["result"],
                    "Newton(x*)": newton_result["result"],
                    "|L - N|": delta,
                }
            ]
        )
        print_table("Контрольная погрешность между формами", comparison)


def task_6() -> None:
    _, _, table = task_5_and_6_tables()
    result = pretty_solve_cubic_splines_table(table)

    section("Задание 6. Кубические сплайны по таблице задания 5")
    task_brief(
        task="Построить кубические сплайны дефекта 1 на каждом отрезке таблицы задания 5.",
        condition="Для отрезков [x_{i-1}, x_i] используется таблица из задания 5.",
        restrictions=(
            "Граничное условие: S''(x_1) = S''(x_6).",
            "В текущей реализации строится натуральный сплайн, поэтому c[0] = c[-1] = 0.",
        ),
        answers=(
            f"Статус построения: {result.get('status')}.",
            f"Коэффициенты [a_i, b_i, c_i, d_i] по интервалам: {result_value(result, 'coefficients')}.",
            f"S''(x_1) = {result_norm(result, 's2_left')}, "
            f"S''(x_6) = {result_norm(result, 's2_right')}, "
            f"разность = {result_norm(result, 's2_diff')}.",
        ),
    )

    details()
    print_result(
        "Кубический сплайн дефекта 1",
        result,
        values=("coefficients", "s2_left", "s2_right", "s2_diff"),
        tables=("h_delta_table", "system_table", "coefficients_table", "spline_table"),
    )


def task_7() -> None:
    x = [0.0, 0.12, 0.19, 0.35, 0.4, 0.45, 0.62, 0.71, 0.84, 0.91, 1.0]
    y = [-1.0, -1.0, -0.9, -0.5, -0.7, -0.6, -0.3, -0.5, 0.4, 0.8, 1.2]
    table = make_xy_table(x, y)
    linear_result = pretty_solve_linear_mls_table(table)
    quadratic_result = pretty_solve_quadratic_mls_table(table)

    section("Задание 7. Метод наименьших квадратов")
    task_brief(
        task="Аппроксимировать таблицу линейным и квадратичным многочленом методом наименьших квадратов.",
        condition="Таблица из задания 7: 11 точек x_i, y_i.",
        restrictions=(
            "Линейная модель: P1(x) = a*x + b.",
            "Квадратичная модель: P2(x) = a*x^2 + b*x + c.",
        ),
        answers=(
            f"P1(x) = {number_inline(linear_result['a'])}*x + {number_inline(linear_result['b'])}, "
            f"max|ошибка| = {result_norm(linear_result, 'residuals_norm_inf')}.",
            f"P2(x) = {number_inline(quadratic_result['a'])}*x^2 + "
            f"{number_inline(quadratic_result['b'])}*x + {number_inline(quadratic_result['c'])}, "
            f"max|ошибка| = {result_norm(quadratic_result, 'residuals_norm_inf')}.",
        ),
    )

    details()
    print_table("Узлы", readable_xy_table(x, y))
    print_result(
        "A) Линейный многочлен",
        linear_result,
        values=("a", "b", "residuals_norm_inf"),
    )
    print_result(
        "B) Квадратичный многочлен",
        quadratic_result,
        values=("a", "b", "c", "residuals_norm_inf"),
    )

    if linear_result["status"] == "ok" and quadratic_result["status"] == "ok":
        comparison = pd.DataFrame(
            {
                "x": x,
                "y": y,
                "P1(x)": linear_result["y_pred"],
                "|y-P1|": linear_result["residuals"],
                "P2(x)": quadratic_result["y_pred"],
                "|y-P2|": quadratic_result["residuals"],
            }
        )
        print_table("Сравнение аппроксимаций", comparison)


def task_8() -> None:
    x, y, table = task_5_and_6_tables()
    x_2 = x[1]
    x_dots = np.array([X_STAR, x_2], dtype=float)
    first_result = pretty_solve_first_diff_table(table, x_dots)
    second_result = pretty_solve_second_diff_table(table, x_dots)
    selected_points = local_quadratic_derivatives(x, y, {"x*": X_STAR, "x_2": x_2})
    x_star_first = selected_points.iloc[0]["y'"]
    x_star_second = selected_points.iloc[0]["y''"]
    x_2_first = selected_points.iloc[1]["y'"]
    x_2_second = selected_points.iloc[1]["y''"]

    section("Задание 8. Первая и вторая производные по таблице задания 5")
    task_brief(
        task="Найти значения первой и второй производных в точках x* и x_2.",
        condition="Используется таблица задания 5.",
        restrictions=(
            f"x* = {X_STAR}.",
            f"x_2 = {x_2}.",
            "Для явного ответа в этих точках дополнительно используется локальный квадратичный многочлен по ближайшим 3 узлам.",
        ),
        answers=(
            f"В x*: y' = {number_inline(x_star_first)}, y'' = {number_inline(x_star_second)}.",
            f"В x_2: y' = {number_inline(x_2_first)}, y'' = {number_inline(x_2_second)}.",
        ),
    )

    details()
    print_result(
        "A) Первая производная: pretty_solve_first_diff_table",
        first_result,
        values=("result", "uniform_grid"),
        tables=("diff_table",),
    )
    print_result(
        "B) Вторая производная: pretty_solve_second_diff_table",
        second_result,
        values=("result", "uniform_grid"),
        tables=("diff_table",),
    )
    print_table("Значения в нужных точках по локальному квадратичному многочлену", selected_points)


def task_9() -> None:
    def f(x: float) -> float:
        return math.sqrt(x) * math.cos(x**2)

    result = pretty_solve_simpson_integration_func(f, 0.0, 1.0, 1e-4)

    section("Задание 9. Интеграл методом Симпсона")
    task_brief(
        task="Вычислить определенный интеграл методом Симпсона.",
        condition="Integral from 0 to 1 of sqrt(x) * cos(x^2) dx.",
        restrictions=(
            "Пределы интегрирования: a = 0, b = 1.",
            "Точность: eps = 1e-4.",
        ),
        answers=(f"Интеграл = {result_value(result, 'result')}, n = {result.get('n', '-')}.",),
    )

    details()
    print_result(
        "Integral int_0^1 sqrt(x) * cos(x^2) dx",
        result,
        values=("result", "a", "b", "eps", "iterations", "n"),
        tables=("iteration_table",),
    )


def cauchy_rhs(x: float, y: float) -> float:
    return 1 - math.sin(x + y) + 0.5 * y / (x + 2)


def task_10() -> None:
    h = 0.1
    x_0 = 0.0
    y_0 = 0.0
    x_star = 1.0
    rk_result = pretty_solve_runge_cutta_func(cauchy_rhs, h, x_0, y_0, x_star)
    adams_result = pretty_solve_adams_func(cauchy_rhs, h, x_0, y_0, x_star)

    section("Задание 10. Задача Коши")
    task_brief(
        task="Решить задачу Коши методами Рунге-Кутты 4-го порядка и Адамса.",
        condition="y' = 1 - sin(x + y) + 0.5*y/(x + 2), y(0) = 0.",
        restrictions=(
            f"Шаг: h = {h}.",
            f"Точность из задания: eps = {EPS}.",
            f"Ответ выводится в контрольной точке x = {x_star}.",
            "Для метода Адамса начальный участок определяется методом Рунге-Кутты.",
        ),
        answers=(
            f"Рунге-Кутта 4: y({x_star}) = {result_value(rk_result, 'result')}.",
            f"Адамс: y({x_star}) = {result_value(adams_result, 'result')}.",
        ),
    )

    details()
    print_result(
        "A) Метод Рунге-Кутты 4-го порядка",
        rk_result,
        values=("result", "h", "x_0", "y_0", "x", "y"),
        tables=("step_table", "interpolation_table"),
    )
    print_result(
        "B) Метод Адамса",
        adams_result,
        values=("result", "h", "x_0", "y_0", "x", "y"),
        tables=("step_table", "interpolation_table"),
    )


def task_11() -> None:
    h = 0.1
    a, b = 1.0, 1.4
    alpha = np.array([1.0, -0.5, 2.0], dtype=float)
    beta = np.array([0.0, 1.0, 4.0], dtype=float)

    def p(x: float) -> float:
        return -1.0

    def q(x: float) -> float:
        return 2.0 / x

    def f(x: float) -> float:
        return x + 0.4

    result = pretty_solve_progonka_func(h, p, q, f, a, b, alpha, beta)

    section("Задание 11. Краевая задача для ОДУ методом прогонки")
    task_brief(
        task="Решить краевую задачу для ОДУ методом прогонки.",
        condition="y'' - y' + 2y/x = x + 0.4; y(1) - 0.5*y'(1) = 2; y'(1.4) = 4.",
        restrictions=(
            f"Шаг: h = {h}.",
            "Порядок точности разностной схемы: O(h^2), как указано в задании.",
            f"Отрезок: [{a}, {b}].",
        ),
        answers=(
            f"x = {result_value(result, 'x')}.",
            f"y = {result_value(result, 'y')}.",
            f"||r||inf = {result_norm(result, 'residuals_norm_inf')}.",
        ),
    )

    details()
    print_result(
        "Метод прогонки для краевой задачи",
        result,
        values=("x", "y", "h", "residuals", "residuals_norm_inf"),
        tables=("grid_table", "boundary_table", "equation_table", "forward_pass_table"),
    )


def main() -> None:
    section("Численные методы: запуск значений из задания")
    line("Порядок вывода", "сначала Задача -> Условие -> Ограничения -> Ответ, затем таблицы")
    line("eps для задач с точностью 0.01", EPS)
    task_1()
    task_2()
    task_3()
    task_4()
    task_5()
    task_6()
    task_7()
    task_8()
    task_9()
    task_10()
    task_11()


if __name__ == "__main__":
    main()
